import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product
import copy


class TrustRegionSymmetric:
    def __init__(
            self,
            eps=1e-3,
            tau_0=0.75,
            tau_1=0.25,
            tau_2=2.,
            tau_3=0.5,
            tau_4=0.5,
            tau_5=2.,
            device='cpu',
            use_std=True,
            probabilistic=False,
            **kwargs
    ):
        self._eps = eps
        self._tau_0 = tau_0
        self._tau_1 = tau_1
        self._tau_2 = tau_2
        self._tau_3 = tau_3
        self._tau_4 = tau_4
        self._tau_5 = tau_5
        self._use_std = use_std
        self._uncessessfull_trials = 0.
        self._probabilistic = probabilistic
        self._device = device

    def step(
            self,
            y_model,
            model,
            previous_psi,
            step,
            optimizer,
            X_data,
            y_data,
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
        used_samples = 2
        X = []
        y = []

        optimizer.update(oracle=model, x=previous_psi, step=step)
        current_psi, status, history = optimizer.optimize()

        func_prev_surr = model.func(previous_psi, num_repetitions=int((1 + self._uncessessfull_trials) * num_repetitions)).detach().clone().to(self._device)
        func_curr_surr = model.func(current_psi, num_repetitions=int((1 + self._uncessessfull_trials) * num_repetitions)).detach().clone().to(self._device)
        # y_model.generate()
        func_prev, X_prev = y_model.generate_data_at_point(n_samples_per_dim=num_repetitions, current_psi=previous_psi)
        func_curr, X_curr = y_model.generate_data_at_point(n_samples_per_dim=num_repetitions, current_psi=current_psi)
        X.append(X_prev); y.append(func_prev); X.append(X_curr); y.append(func_curr)

        func_prev = y_model.loss(func_prev)
        func_curr = y_model.loss(func_curr)
        std_raw = (
                (
                        func_prev.std() / (len(func_prev) - 1)**(0.5) + func_curr.std() / (len(func_curr) - 1)**(0.5)
                )
        ).item()

        # correction to garantee convergance to zero of step
        std = std_raw / (1 + self._uncessessfull_trials)
        if self._use_std:
            f_cut = std_raw / (1 + self._uncessessfull_trials)**2
        else:
            f_cut = self._eps
        func_prev = func_prev.mean()
        func_curr = func_curr.mean()
        rho = ((func_prev - func_curr) / (func_prev_surr - func_curr_surr)).item()
        success = False
        if rho > self._eps and (func_prev - func_curr).item() >= f_cut:
            psi = current_psi
            success = True
            self._uncessessfull_trials = max(0, self._uncessessfull_trials - 1)
        else:
            psi = previous_psi
            self._uncessessfull_trials += 1

        step_status = None
        # --------------------------------------------------------------------------------
        if rho > self._tau_0:
            if success:
                step_status = "Full success"
                if self._probabilistic:
                    min_ = step
                    max_ = step * self._tau_2
                    step = np.random.uniform(low=min_, high=max_)
                else:
                    step = self._tau_2 * step

            else:
                if (func_prev - func_curr).abs().item() <= std and (current_psi - previous_psi).norm().item() > step * 1e-2:
                    if self._probabilistic:
                        min_ = step
                        max_ = step * self._tau_2
                        step = np.random.uniform(low=min_, high=max_)
                    else:
                        step = self._tau_2 * step
                else:
                    if self._probabilistic:
                        min_ = step
                        max_ = step * self._tau_5
                        step = np.random.uniform(low=min_, high=max_)
                    else:
                        step = self._tau_5 * step
        # --------------------------------------------------------------------------------
        elif self._tau_0 > rho > self._tau_1:
            step = step
        # --------------------------------------------------------------------------------
        elif rho < self._tau_1:
            if success:
                if self._probabilistic:
                    min_ = self._tau_4 * step
                    max_ = step
                    step = np.random.uniform(low=min_, high=max_)
                else:
                    step = self._tau_4 * step
            else:
                if self._probabilistic:
                    min_ = self._tau_3 * step
                    max_ = step
                    step = np.random.uniform(low=min_, high=max_)
                else:
                    step = self._tau_3 * step

        # last correction
        psi_uniques = X_data[:, :len(psi)].unique(dim=0)
        ys = []
        y_data = y_model.loss(y_data)
        for psi_unique in psi_uniques:
            mask = (X_data[:, :len(psi)] == psi_unique).all(dim=1)
            ys.append(y_data[mask].mean().item())

        if np.std(ys) < std_raw:
            step = self._tau_2 * step

        return psi, step, optimizer, {
            'diff_real': (func_prev - func_curr).item(),
            'diff_surrogate': (func_prev_surr - func_curr_surr).item(),
            'rho': rho,
            'success': success,
            'proposed_psi': current_psi.detach().cpu().numpy(),
            'grad': -model.grad(previous_psi, num_repetitions=num_repetitions).detach().cpu().numpy(),
            'grad_corrected': (current_psi - previous_psi).detach().cpu().numpy(),
            'step_status': step_status,
            'X': torch.cat(X, dim=0),
            'y': torch.cat(y, dim=0),
            'uncessessfull_trials': self._uncessessfull_trials,
            'used_samples': used_samples
        }
