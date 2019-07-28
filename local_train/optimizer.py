from abc import ABC, abstractmethod
import numpy as np
from base_model import BaseConditionalGenerationOracle
from numpy.linalg import LinAlgError
from line_search_tool import LineSearchTool, get_line_search_tool
from logger import BaseLogger
from collections import defaultdict
import scipy
import matplotlib.pyplot as plt
import torch
import time
SUCCESS = 'success'
ITER_ESCEEDED = 'iterations_exceeded'
COMP_ERROR = 'computational_error'


class BaseOptimizer(ABC):
    """
    Base class for optimization of some function with logging
    functionality spread by all classes
    """
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 tolerance: torch.Tensor = torch.tensor(1e-4),
                 trace: bool = True,
                 num_repetitions: int = 1000,
                 max_iters: int = 1000,
                 *args, **kwargs):
        self._oracle = oracle
        self._history = defaultdict(list)
        self._x = x
        self._tolerance = tolerance
        self._trace = trace
        self._max_iters = max_iters
        self._num_repetitions = num_repetitions
        self._num_iter = 0

    def _update_history(self, init_time):
        self._history['time'].append(
            time.time() - init_time
        )
        self._history['func'].append(
            self._oracle.func(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )
        self._history['grad'].append(
            self._oracle.grad(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )
        self._history['x'].append(
            self._x.detach().cpu().numpy()
        )

    def optimize(self):
        for i in range(self._max_iters):
            status = self._step()
            if status == COMP_ERROR:
                return self._x.detach().clone(), status, self._history
            elif status == SUCCESS:
                return self._x.detach().clone(), status, self._history
        return self._x.detach().clone(), ITER_ESCEEDED, self._history

    @abstractmethod
    def _step(self):
        """
        Compute update of optimized parameter
        :return:
        """
        raise NotImplementedError('_step is not implemented.')

    def _post_step(self, init_time):
        self._num_iter += 1
        if self._trace:
            self._update_history(init_time=init_time)


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._alpha_k = None
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        d_k = -self._oracle.grad(x_k, num_repetitions=self._num_repetitions)

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)
        print(self._alpha_k)
        with torch.no_grad():
            x_k = x_k + d_k * self._alpha_k
        grad_norm = torch.norm(d_k).item()
        self._x = x_k

        super()._post_step(init_time)
        # seems not cool to call super method in the middle of function...

        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class NewtonOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr  # in newton method learning rate used to initialize line search tool
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)


    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        d_k = -self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        h_d = self._oracle.hessian(x_k, num_repetitions=self._num_repetitions)
        try:
            c_and_lower = scipy.linalg.cho_factor(h_d.detach().cpu().numpy())
            d_k = scipy.linalg.cho_solve(c_and_lower, d_k.detach().cpu().numpy())
            d_k = torch.tensor(d_k).float().to(self._oracle.device)
        except LinAlgError:
            pass
        alpha_k = self._line_search_tool.line_search(self._oracle,
                                                     x_k,
                                                     d_k,
                                                     previous_alpha=self._lr,
                                                     num_repetitions=self._num_repetitions)
        with torch.no_grad():
            x_k = x_k + d_k * alpha_k
        self._x = x_k
        super()._post_step(init_time)

        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


def d_computation_in_lbfgs(d, history):
    l = len(history)
    mu = list()
    for i in range(l)[::-1]:
        s = history[i][0]
        y = history[i][1]
        mu.append(s.dot(d) / s.dot(y))
        d -= y * mu[-1]
    mu = mu[::-1]
    s = history[-1][0]
    y = history[-1][1]
    d = d * s.dot(y) / y.dot(y)
    for i in range(l):
        s = history[i][0]
        y = history[i][1]
        beta = y.dot(d) / s.dot(y)
        d += (mu[i] - beta) * s
    return d


class LBFGSOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 memory_size: int = 10,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._sy_history = list()
        self._alpha_k = None
        self._memory_size = memory_size
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        init_time = time.time()

        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)

        if len(self._sy_history) > 0:
            d_k = d_computation_in_lbfgs(-g_k.clone().detach(), self._sy_history)
        else:
            d_k = - g_k.clone().detach()

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)

        x_k = x_k + d_k * self._alpha_k
        self._x = x_k.clone().detach()
        g_k_new = self._oracle.grad(x_k)
        self._sy_history.append((self._alpha_k * d_k, g_k_new - g_k))
        if len(self._sy_history) > self._memory_size:
            self._sy_history.pop(0)

        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class ConjugateGradientsOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._alpha_k = None
        self._d_k = None
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        init_time = time.time()

        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        if self._d_k is None:
            self._d_k = -g_k.clone().detach()

        norm_squared = g_k.pow(2).sum()

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               self._d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               self._d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)
        # TODO: dirty hack, what to do when line_search_tool is not converged?
        if self._alpha_k is None:
            self._alpha_k = self._lr

        x_k = x_k + self._d_k * self._alpha_k
        g_k_next = self._oracle.grad(x_k)
        beta_k = g_k_next.dot((g_k_next - g_k)) / norm_squared
        self._d_k = -g_k_next + beta_k * self._d_k
        self._x = x_k.clone().detach()

        super()._post_step(init_time)
        grad_norm = torch.norm(g_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(self._d_k).all()):
            return COMP_ERROR