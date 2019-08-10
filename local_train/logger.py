from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import auc
from scipy.spatial.distance import cosine
from prd_score import compute_prd_from_embedding
import lhsmdu
import numpy as np
import torch
import sys
sys.path.append('../')
from utils import generate_local_data_lhs
from tqdm import tqdm
from model import YModel
from pyDOE import lhs
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

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
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
        self._optimizer_logs['alpha'].extend(history['alpha'])
        return None

    @staticmethod
    def calc_grad_metric_in_point(oracle, y_sampler, psi, num_repetitions):
        grad_true = y_sampler.grad(psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        grad_fake = oracle.grad(psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        return cosine(grad_true, grad_fake)

    @staticmethod
    def calc_func_metric_in_point(oracle, y_sampler, psi, num_repetitions):
        y_true = y_sampler.func(psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        y_fake = oracle.func(psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        return np.abs((y_true - y_fake) / y_true)

    @staticmethod
    def calc_hessian_metric_in_point(oracle, y_sampler, psi, num_repetitions):
        hessian_true = y_sampler.hessian(psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        hessian_fake = oracle.hessian(psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        eigvecs_true, eigvals_true, _ = np.linalg.svd(hessian_true)
        eigvecs_fake, eigvals_fake, _ = np.linalg.svd(hessian_fake)

        cosine_distane_hessian = []
        for i in range(eigvecs_true.shape[1]):
            cosine_distane_hessian.append(
                cosine(eigvecs_true[:, i], eigvecs_fake[:, i])
            )
        return np.abs((eigvals_true - eigvals_fake) / eigvals_true).tolist(), cosine_distane_hessian

    @abstractmethod
    def log_oracle(self,
                   oracle,
                   y_sampler,
                   current_psi: torch.Tensor,
                   step_data_gen: float,
                   scale_step: int = 2,
                   num_samples: int = 1000,
                   num_repetitions: int = 2000,
                   calc_hessian_metrics: bool = False):
        """

        :param oracle:
        :param y_sampler:
        :param current_psi:
        :param step_data_gen:
        :param num_samples:
        :param scale_step:
        :param num_repetitions:
        :return:
            dict of:
                relative difference of values of loss function
                cosine distance of gradients inside of training region
                cosine distance of gradients outside of training region
                relative difference of eigen values of gessian inside of training region
                cosine similarity of eigen vectors of gessina outside of training region
                PRD score of generated samples inside of training region
                PRD score of generated samples outside of training region
        """
        metrics = defaultdict(list)
        num_samples = 90
        psi_dim = current_psi.shape[0]  # y_sampler._psi_dim
        psis_inside = torch.tensor(lhs(len(current_psi), num_samples)).float().to(current_psi.device)
        psis_inside = step_data_gen * (psis_inside * 2 - 1) + current_psi.view(1, -1)

        psis_outside = torch.tensor(lhs(len(current_psi), num_samples)).float().to(current_psi.device)
        psis_outside = scale_step * step_data_gen * (psis_outside * 2 - 1) + current_psi.view(1, -1)

        psis = torch.cat([psis_inside, psis_outside], dim=0)
        for i, psi in tqdm(enumerate(psis)):
            if (psi - current_psi).norm().item() < step_data_gen:
                metrics["grad_metric_inside"].append(
                    self.calc_grad_metric_in_point(oracle=oracle, y_sampler=y_sampler,
                                                   psi=psi, num_repetitions=num_repetitions)
                )
                metrics["func_metric_inside"].append(
                    self.calc_func_metric_in_point(oracle=oracle, y_sampler=y_sampler,
                                                   psi=psi, num_repetitions=num_repetitions)
                )
                if calc_hessian_metrics:
                    eigenvalues_distances_hess, eigenvectors_distanes_hess = self.calc_hessian_metric_in_point(
                        oracle=oracle,
                        y_sampler=y_sampler,
                        psi=psi,
                        num_repetitions=num_repetitions)
                    metrics["eigenvalues_metric_inside"].extend(eigenvalues_distances_hess)
                    metrics["eigenvectrors_metric_inside"].extend(eigenvectors_distanes_hess)
            else:
                metrics["grad_metric_outside"].append(
                    self.calc_grad_metric_in_point(oracle=oracle, y_sampler=y_sampler,
                                                   psi=psi, num_repetitions=num_repetitions)
                )
                metrics["func_metric_outside"].append(
                    self.calc_func_metric_in_point(oracle=oracle, y_sampler=y_sampler,
                                                   psi=psi, num_repetitions=num_repetitions)
                )
                if calc_hessian_metrics:
                    eigenvalues_distances_hess, eigenvectors_distanes_hess = self.calc_hessian_metric_in_point(
                        oracle=oracle,
                        y_sampler=y_sampler,
                        psi=psi,
                        num_repetitions=num_repetitions)
                    metrics["eigenvalues_metric_outside"].extend(eigenvalues_distances_hess)
                    metrics["eigenvectrors_metric_outside"].extend(eigenvectors_distanes_hess)

        data, conditions = y_sampler.generate_local_data_lhs(
            n_samples_per_dim=100,
            step=step_data_gen,
            current_psi=current_psi,
            n_samples=100
        )
        data_real = torch.cat([data, conditions], dim=1).detach().cpu().numpy()
        data_fake = oracle.generate(conditions)
        data_fake = torch.cat([data_fake, conditions], dim=1).detach().cpu().numpy()
        print(data_fake.shape, data_real.shape)
        print("PRD inside")
        precision_inside, recall_inside = compute_prd_from_embedding(data_real, data_fake)
        metrics["precision_inside"].extend(precision_inside.tolist())
        metrics["recall_inside"].extend(recall_inside.tolist())

        data, conditions = y_sampler.generate_local_data_lhs(
            n_samples_per_dim=100,
            step=scale_step * step_data_gen,
            current_psi=current_psi,
            n_samples=100
        )
        data_real = torch.cat([data, conditions], dim=1).detach().cpu().numpy()
        data_fake = oracle.generate(conditions)
        data_fake = torch.cat([data_fake, conditions], dim=1).detach().cpu().numpy()
        print("PRD outside")
        precision_outside, recall_outside = compute_prd_from_embedding(data_real, data_fake)
        metrics["precision_outside"].extend(precision_outside.tolist())
        metrics["recall_outside"].extend(recall_outside.tolist())
        return metrics

    @abstractmethod
    def log_performance(self, y_sampler, current_psi):
        self._perfomance_logs['time'].append(time.time() - self._time)
        self._time = time.time()
        print(current_psi)
        self._perfomance_logs['func'].append(y_sampler.func(current_psi, num_repetitions=5000).detach().cpu().numpy())
        self._perfomance_logs['psi'].append(current_psi.detach().cpu().numpy())
        self._perfomance_logs['psi_grad'].append(y_sampler.grad(current_psi, num_repetitions=5000).detach().cpu().numpy())


class SimpleLogger(BaseLogger):
    def __init__(self):
        super(SimpleLogger, self).__init__()

    @staticmethod
    def _print_num_nans_in_metric(metrics, metric_name):
        print("{} contains {} inf/nans".format(
            metric_name, (~np.isfinite(metrics[metric_name])).sum()
        ))

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
            axs[0][1].plot(xs[:, i])
        axs[0][1].grid()
        axs[0][1].set_ylabel("$\mu$", fontsize=19)
        axs[0][1].set_xlabel("iter", fontsize=19)
        axs[0][1].plot(np.linalg.norm(xs, axis=1),
                       c='k', linewidth=2,
                       label='$| \mu |$')
        axs[0][1].set_title("Norm: {}".format(np.linalg.norm(xs[-1, :])))

        alpha_k = np.array(self._optimizer_logs['alpha_k'])
        axs[1][0].plot(alpha_k)
        axs[1][0].grid()
        axs[1][0].set_ylabel("alpha_k", fontsize=19)
        axs[1][0].set_xlabel("iter", fontsize=19)

        ds = np.array(self._optimizer_logs['grad'])
        for i in range(ds.shape[1]):
            axs[1][1].plot(ds[:, i])
        axs[1][1].grid()
        axs[1][1].set_ylabel("$\delta \mu$", fontsize=19)
        axs[1][1].plot(np.linalg.norm(ds, axis=1),
                       c='k', linewidth=2,
                       label='$| \delta \mu |$')
        axs[1][1].set_xlabel("iter", fontsize=19)
        axs[1][1].set_title("Norm: {}".format(np.linalg.norm(ds[-1, :])))

        figure.legend()
        return figure

    def log_oracle(self, oracle, y_sampler,
                   current_psi,
                   step_data_gen,
                   scale_step=2,
                   num_samples=1000,
                   num_repetitions=2000):
        metrics = super().log_oracle(oracle=oracle, y_sampler=y_sampler,
                                    current_psi=current_psi, step_data_gen=step_data_gen,
                                    scale_step=scale_step, num_samples=num_samples,
                                    num_repetitions=num_repetitions)

        figure, axs = plt.subplots(5, 2, figsize=(18, 8 * 5), dpi=300)
        self._print_num_nans_in_metric(metrics, "func_metric_inside")
        axs[0][0].hist(metrics["func_metric_inside"], bins=50, density=True)
        axs[0][0].grid()
        axs[0][0].set_ylabel("Loss relative error inside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "func_metric_outside")
        axs[0][1].hist(metrics["func_metric_outside"], bins=50, density=True)
        axs[0][1].grid()
        axs[0][1].set_ylabel("Loss relative error outside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "grad_metric_inside")
        axs[1][0].hist(metrics["grad_metric_inside"], bins=50, density=True)
        axs[1][0].grid()
        axs[1][0].set_ylabel("Cosine distance of gradients inside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "grad_metric_outside")
        axs[1][1].hist(metrics["grad_metric_outside"], bins=50, density=True)
        axs[1][1].grid()
        axs[1][1].set_ylabel("Cosine distance of gradients outside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "eigenvalues_metric_inside")
        axs[2][0].hist(metrics["eigenvalues_metric_inside"], bins=50, density=True, range=(0, 5))
        axs[2][0].grid()
        axs[2][0].set_ylabel("Hessian eigenvalues relative error inside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "eigenvalues_metric_outside")
        axs[2][1].hist(metrics["eigenvalues_metric_outside"], bins=50, density=True, range=(0, 5))
        axs[2][1].grid()
        axs[2][1].set_ylabel("Hessian eigenvalues relative error outside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "eigenvectrors_metric_inside")
        axs[3][0].hist(metrics["eigenvectrors_metric_inside"], bins=50, density=True)
        axs[3][0].grid()
        axs[3][0].set_ylabel("Cosine distance of hessian eigenvectors inside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "eigenvectrors_metric_outside")
        axs[3][1].hist(metrics["eigenvectrors_metric_outside"], bins=50, density=True)
        axs[3][1].grid()
        axs[3][1].set_ylabel("Cosine distance of hessian eigenvectors outside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "precision_inside")
        self._print_num_nans_in_metric(metrics, "recall_inside")
        axs[4][0].step(metrics["precision_inside"], metrics["recall_inside"])
        axs[4][0].grid()
        axs[4][0].set_ylabel("PRD score inside", fontsize=19)

        self._print_num_nans_in_metric(metrics, "precision_outside")
        self._print_num_nans_in_metric(metrics, "recall_outside")
        axs[4][1].step(metrics["precision_outside"], metrics["recall_outside"])
        axs[4][1].grid()
        axs[4][1].set_ylabel("PRD score outside", fontsize=19)

        return metrics, figure

    def log_performance(self, y_sampler, current_psi):
        super().log_performance(y_sampler=y_sampler, current_psi=current_psi)


class CometLogger(SimpleLogger):
    def __init__(self, experiment):
        super(CometLogger, self).__init__()
        self._experiment = experiment
        self._epoch = 0

    def log_optimizer(self, optimizer):
        figure = super().log_optimizer(optimizer)
        self._experiment.log_figure("Optimization dynamic", figure, overwrite=True)

    def log_oracle(self, oracle, y_sampler,
                   current_psi,
                   step_data_gen,
                   scale_step=2,
                   num_samples=1000,
                   num_repetitions=2000):
        metrics, figure = super().log_oracle(oracle=oracle, y_sampler=y_sampler,
                                             current_psi=current_psi, step_data_gen=step_data_gen,
                                             scale_step=scale_step, num_samples=num_samples,
                                             num_repetitions=num_repetitions)

        self._experiment.log_figure("Oracle state", figure, overwrite=True)
        self._experiment.log_metric('PRD inside', auc(metrics["precision_inside"], metrics["recall_inside"]), step=self._epoch)
        self._experiment.log_metric('PRD outside', auc(metrics["precision_outside"], metrics["recall_outside"]), step=self._epoch)

    def log_performance(self, y_sampler, current_psi):
        super().log_performance(y_sampler=y_sampler, current_psi=current_psi)
        self._experiment.log_metric('Time spend', self._perfomance_logs['time'][-1], step=self._epoch)
        self._experiment.log_metric('Func value', self._perfomance_logs['func'][-1], step=self._epoch)
        psis = self._perfomance_logs['psi'][-1]
        self._experiment.log_metric('Psi norm', np.linalg.norm(psis), step=self._epoch)
        # for i, psi in enumerate(psis):
        # self._experiment.log_metric('Psi_{}'.format(i), psi, step=self._epoch)

        psi_grad = self._perfomance_logs['psi_grad'][-1]
        self._experiment.log_metric('Psi grad norm', np.linalg.norm(psi_grad), step=self._epoch)
        self._epoch += 1


class GANLogger(object):
    def __init__(self, experiment):
        self._experiment = experiment
        self._epoch = 0

    def log_losses(self, losses):
        self._experiment.log_metric("d_loss", np.mean(losses[0]), step=self._epoch)
        self._experiment.log_metric("g_loss", np.mean(losses[1]), step=self._epoch)
        self._epoch += 1

