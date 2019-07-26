from abc import ABC, abstractmethod
import numpy as np
from base_model import BaseConditionalGenerationOracle
from logger import BaseLogger
from collections import defaultdict
import matplotlib.pyplot as plt
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
                 trace: bool = False,
                 num_repetitions: int = 1000,
                 max_iters: int = 1000,
                 logger: BaseLogger = None,
                 *args, **kwargs):
        self._oracle = oracle
        self._history = defaultdict(list)
        self._x = x
        self._tolerance = tolerance
        self._trace = trace
        self._max_iters = max_iters
        self._num_repetitions = num_repetitions
        self._logger = logger
        self._num_iter = 0

    def update_history(self, init_time):
        self._history['time'].append(
            time.time() - init_time
        )
        self._history['func'].append(
            self._oracle.func(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )
        self._history['grad'].append(
            self._oracle.grad(self._x, num_repetitions=self._num_repetitions).detach().cpu().numpy()
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
        raise NotImplementedError('_step is not implemented.')

    def _post_step(self, init_time):
        self._num_iter += 1
        if trace:
            self._update_history(init_time=init_time)
        if self._logger:
            self._logger.log(self)


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k)
        d_k = -self._oracle.grad(x_k)
        with torch.no_grad():
            x_k = x_k + d_k * self._lr
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite().all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR

        self._x = x_k
        # seems not cool to call super method at the end of function...
        super()._post_step(init_time)


class NewtonOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1., # in newton learning rate should == 1 usually
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k)
        d_k = -self._oracle.grad(x_k)
        h_d = self._oracle.hessian(x_k)
        try:
            c_and_lower = scipy.linalg.cho_factor(h_d.detach().cpu().numpy())
            d_k = scipy.linalg.cho_solve(c_and_lower, d_k.detach().cpu().numpy())
            d_k = torch.tensor(d_k).float().to(self._oracle.device)
        except:
            pass

        with torch.no_grad():
            x_k = x_k + d_k * self._lr

        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite().all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR

        self._x = x_k
        # seems not cool to call super method at the end of function...
        super()._post_step(init_time)