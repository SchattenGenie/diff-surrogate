from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import cosine
import numpy as np
import torch
import sys
sys.path.append('../')
from utils import generate_local_data_lhs
from model import YModel
import time


# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y
    """
    window_len = min(window_len, len(x) - 1)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


class BaseLogger(ABC):
    def __init__(self):
        self._optimizer_logs = defaultdict(list)
        self._oracle_logs = defaultdict(list)
        self._perfomance_logs = defaultdict(list)
        self._time = time.time()

    @abstractmethod
    def log_optimizer(self, optimizer):
        history = optimizer._history
        self._optimizer_logs['x'].extend(history['x'])
        self._optimizer_logs['func'].extend(history['func'])
        self._optimizer_logs['grad'].extend(history['grad'])
        self._optimizer_logs['time'].extend(history['time'])
        return None

    @abstractmethod
    def log_oracle(self, oracle, y_sampler, current_psi):
        # TODO: somehow refactor
        # because its very nasty
        psi_dim = current_psi # y_sampler._psi_dim
        data, conditions = y_sampler.generate_local_data_lhs(
            n_samples_per_dim=3000,
            step=1,
            current_psi=current_psi,
            n_samples=30
        )
        # TODO: even more nasty things
        losses_oracle = oracle.func(conditions[:, :psi_dim]).detach().cpu().numpy()
        grads_oracle = oracle.grad(conditions[:, :psi_dim]).detach().cpu().numpy()
        losses_real = y_sampler.func(conditions).detach().cpu().numpy()
        grads_real = y_sampler.grad(conditions).detach().cpu().numpy()[:, :psi_dim]

        conditions = conditions.detach().cpu().numpy()
        unique_conditions = np.unique(conditions[:, :psi_dim], axis=0)
        losses_dist = []
        grads_dist = []
        for condition in unique_conditions:
            mask = (conditions[:, :psi_dim] == condition).all(axis=1)
            losses_dist.append(losses_oracle[mask].mean() - losses_real[mask].mean())
            grads_dist.append(cosine(grads_oracle[mask].mean(axis=0), grads_real[mask].mean(axis=0)))

        return losses_dist, grads_dist

    @abstractmethod
    def log_performance(self, y_sampler, current_psi):
        self._perfomance_logs['time'].append(time.time() - self._time)
        self._time = time.time()
        print(current_psi)
        self._perfomance_logs['func'].append(y_sampler.func(current_psi, num_repetitions=5000))
        self._perfomance_logs['psi'].append(current_psi.detach().cpu().numpy())
        self._perfomance_logs['psi_grad'].append(y_sampler.grad(current_psi, num_repetitions=5000))

class SimpleLogger(BaseLogger):
    def __init__(self):
        super(SimpleLogger, self).__init__()

    def log_optimizer(self, optimizer):
        super().log_optimizer(optimizer)

        figure, axs = plt.subplots(2, 2, figsize=(18, 18))

        losses = np.array(self._optimizer_logs['func'])
        axs[0][0].plot(losses)
        axs[0][0].grid()
        axs[0][0].set_ylabel("Loss", fontsize=19)
        axs[0][0].set_xlabel("iter", fontsize=19)
        axs[0][0].plot((smooth(losses, window_len=10)), c='r')

        xs = np.array(self._optimizer_logs['x'])
        for i in range(xs.shape[1]):
            axs[0][1].plot(xs[:, i], label=i)
        axs[0][1].grid()
        axs[0][1].set_ylabel("$\mu$", fontsize=19)
        axs[0][1].set_xlabel("iter", fontsize=19)

        times = np.array(self._optimizer_logs['time'])
        axs[1][0].plot(times)
        axs[1][0].grid()
        axs[1][0].set_ylabel("Time spend", fontsize=19)
        axs[1][0].set_xlabel("iter", fontsize=19)

        ds = np.array(self._optimizer_logs['grad'])
        for i in range(ds.shape[1]):
            axs[1][1].plot(ds[:, i], label=i)
        axs[1][1].grid()
        axs[1][1].set_ylabel("$\delta \mu$", fontsize=19)
        axs[1][1].set_xlabel("iter", fontsize=19)

        figure.legend()
        return figure

    def log_oracle(self, oracle, y_sampler, current_psi):
        losses_dist, grads_dist = super().log_oracle(oracle, y_sampler, current_psi)

        figure, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].hist(losses_dist, bins=30, density=True)
        axs[0].grid()
        axs[0].set_ylabel("Loss dist", fontsize=19)

        axs[1].hist(grads_dist, bins=30, density=True)
        axs[1].grid()
        axs[1].set_ylabel("Grads cosine similarity", fontsize=19)

        return figure

    def log_performance(self, y_sampler, current_psi):
        super().log_performance(y_sampler=y_sampler, current_psi=current_psi)


class CometLogger(SimpleLogger):
    def __init__(self, experiment):
        super(CometLogger, self).__init__()
        self._experiment = experiment

    def log_optimizer(self, optimizer):
        figure = super().log_optimizer(optimizer)
        self._experiment.log_figure("Optimization dynamic", figure, overwrite=True)

    def log_oracle(self, oracle, y_sampler, current_psi):
        figure = super().log_oracle(oracle, y_sampler, current_psi)
        self._experiment.log_figure("Oracle state", figure, overwrite=True)

    def log_performance(self, y_sampler, current_psi):
        super().log_performance(y_sampler=y_sampler, current_psi=current_psi)
        self._experiment.log_metric('Time spend', self._perfomance_logs['time'][-1])
        self._experiment.log_metric('Func value', self._perfomance_logs['func'][-1])
        psis = self._perfomance_logs['psi'][-1]
        for i, psi in enumerate(psis):
            self._experiment.log_metric('Psi_{}'.format(i), psi)

        psi_grad = self._perfomance_logs['psi_grad'][-1]
        self._experiment.log_metric('Psi grad norm', psi_grad.norm().item())



