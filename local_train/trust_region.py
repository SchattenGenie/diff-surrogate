import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product


class TrustRegion:
    def __init__(self, device='cpu'):
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
        if rho < 1 / 4:
            step = func_curr_grad.norm().item()
        else:
            if rho > 3 / 4:
                step = step * 2
        if rho > 0.01:
            psi = current_psi
        else:
            psi = previous_psi
        print("trust", rho, psi, step)

        return psi, step