import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product


# TODO: update lr for optimizer as in TrustRegion -- done?
# TODO: try line search for surrogate optimization
# TODO: in TR assumed that (func_prev_surr - func_curr_surr) positive by construction(not anymore!) what to do?
class TrustRegion:
    def __init__(self, tau_0=1e-3, tau_1=2., tau_2=0.25, tau_3=0.25, tau_4=0.5, tau_5=1., device='cpu'):
        self._tau_0 = tau_0
        self._tau_1 = tau_1
        self._tau_2 = tau_2
        self._tau_3 = tau_3
        self._tau_4 = tau_4
        self._tau_5 = tau_5
        self._device = device

    def step(
            self,
            y_model,
            model,
            previous_psi,
            current_psi,
            step,
            optimizer_config,
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
        func_prev_surr = model.func(previous_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
        func_curr_surr = model.func(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)

        # TODO: generate -> loss -> jacknife
        func_prev = y_model.func(previous_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
        func_curr = y_model.func(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)

        rho = (func_prev - func_curr) / (func_prev_surr - func_curr_surr)  # positive both den and ...
        std = np.std(
            [y_model.func(current_psi, num_repetitions=num_repetitions).item() for _ in range(100)]
        )

        success = False
        if rho < self._tau_0 or (func_prev - func_curr) <= std:
            psi = previous_psi
        else:
            psi = current_psi
            success = True

        if rho < self._tau_2:
            # s_norm = (current_psi - previous_psi).norm().item()
            # TODO: std should be inherited from batch data
            step = np.random.uniform(
                low=min(max(std, self._tau_4 * step), step),
                high=step
            )
            # step = np.random.uniform(
            #     low=min(s_norm * self._tau_3, self._tau_4 * step),
            #     high=max(s_norm * self._tau_3, self._tau_4 * step)
            # )
        # TODO: in case of None calculations
        # TODO: check border cases
        elif rho > self._tau_2:
            delta_psi = (previous_psi - current_psi).norm().item()
            # TODO: smth wrong here
            max_ = max((delta_psi + step) * self._tau_1 / 2, std)
            min_ = min(max_, step)
            step = np.random.uniform(low=min_, high=max_)
        # print("TRUST", rho, psi, step)
        optimizer_config['lr'] = self._tau_5 * step
        if hasattr(optimizer, '_base_optimizer'):
            optimizer.lr = optimizer_config['lr']
        if hasattr(optimizer, '_base_optimizer'):
            optimizer._base_optimizer.param_groups[0]['lr'] = optimizer_config['lr']
        return psi, step, {
            'diff_real': (func_prev - func_curr),
            'diff_surrogate': (func_prev_surr - func_curr_surr),
            'success': success
        }
