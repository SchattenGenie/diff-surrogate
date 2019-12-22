import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product


class TrustRegion:
    def __init__(self, tau_0=1e-3, tau_1=2., tau_2=0.25, tau_3=0.25, tau_4=0.5, tdevice='cpu'):
        self._tau_0 = tau_0
        self._tau_1 = tau_1
        self._tau_2 = tau_2
        self._tau_3 = tau_3
        self._tau_4 = tau_4
        self._device = device

    def step(self, y_model, model, previous_psi, current_psi, step, num_repetitions=10000):
        """
        NotaBene: assumed that conditions_grid[-1] is a central point(where we estimate gradients)
        :param model:
        :param conditions_grid:
        :return:
        """
        func_prev_surr = model.func(previous_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
        func_curr_surr = model.func(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
        func_curr_grad = model.grad(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)

        func_prev = y_model.func(previous_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
        func_curr = y_model.func(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)

        rho = (func_prev - func_curr) / (func_prev_surr - func_curr_surr)

        if rho < self._tau_0:
            psi = previous_psi
        else:
            psi = current_psi


        if rho < self._tau_2:
            s_norm = (current_psi - previous_psi).norm().item()
            step = np.random.uniform(
                low=min(s_norm * self._tau_3, self._tau_4 * step),
                high=min(s_norm * self._tau_3, self._tau_4 * step)
            )
        else:
            step = np.random.uniform(low=step, high=self._tau_1 * step)
        print("TRUST", rho, psi, step)

        return psi, step