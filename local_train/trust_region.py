import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product
import copy


# TODO: update lr for optimizer as in TrustRegion -- done?
# TODO: try line search for surrogate optimization
# TODO: in TR assumed that (func_prev_surr - func_curr_surr) positive by construction(not anymore!) what to do?
class TrustRegion:
    def __init__(
            self,
            tau_0=1e-3,
            tau_1=2.,
            tau_2=0.25,
            tau_3=0.25,
            tau_4=0.5,
            tau_5=1.,
            device='cpu',
            **kwargs
    ):
        self._tau_0 = tau_0
        self._tau_1 = tau_1
        self._tau_2 = tau_2
        self._tau_3 = tau_3
        self._tau_4 = tau_4
        self._tau_5 = tau_5
        self._uncessessfull_trials = 0.
        self._device = device

    def step(
            self,
            y_model,
            model,
            previous_psi,
            step,
            optimizer,
            num_repetitions=10000,
    ):
        """
        NotaBene: assumed that conditions_grid[-1] is a central point(where we estimate gradients)
        :param y_model:
        :param model:
        :param previous_psi:
        :param current_psi:
        :param step:
        :param optimizer_config:
        :param optimizer:
        :param num_repetitions:
        :return:
        """
        optimizer.update(oracle=model, x=previous_psi, step=step)
        current_psi, status, history = optimizer.optimize()

        func_prev_surr = model.func(previous_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
        func_curr_surr = model.func(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)

        func_prev = y_model.func(previous_psi.repeat(num_repetitions, 1)).detach().clone().to(self._device)
        func_curr = y_model.func(current_psi.repeat(num_repetitions, 1)).detach().clone().to(self._device)
        std = (
                (
                        func_prev.std() / (len(func_prev) - 1)**(0.5) +
                        func_prev.std() / (len(func_prev) - 1)**(0.5)
                ) / 2.
        ).item()
        func_prev = func_prev.mean()
        func_curr = func_curr.mean()

        rho = ((func_prev - func_curr) / (func_prev_surr - func_curr_surr)).item()

        print(
            "psi diff norm={}".format((current_psi - previous_psi).norm().item()),
            "func surr diff={}".format((func_prev_surr - func_curr_surr).item()),
            "func diff={}".format((func_prev - func_curr).item()),
            "std={}".format(std),
            "dist to opt={}".format((current_psi - 1.).norm().item()),
            previous_psi,
            current_psi,

        )

        success = False
        if rho < self._tau_0 or (func_prev - func_curr).item() <= std:
            psi = previous_psi
        else:
            psi = current_psi
            success = True

        if rho < self._tau_2:
            step = np.random.uniform(
                low=min((self._tau_4 * step + std) / 2., step),
                high=step
            )
        elif rho > self._tau_2:
            optimizer.update_optimizer()
            delta_psi = (previous_psi - current_psi).norm().item()
            # TODO: smth wrong here
            std_var = (std / model.grad(current_psi, num_repetitions=num_repetitions)).pow(2).mean().sqrt().item()
            max_ = max((delta_psi + step) * self._tau_1 / 2, std_var)
            min_ = min(max_, step)
            step = np.random.uniform(low=min_, high=max_)

        return psi, step, optimizer, {
            'diff_real': (func_prev - func_curr),
            'diff_surrogate': (func_prev_surr - func_curr_surr),
            'success': success,
            'grad': model.grad(previous_psi, num_repetitions=num_repetitions).detach().cpu().numpy()
        }
