import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product
import copy


class TrustRegion:
    def __init__(
            self,
            tau_0=1e-3,
            tau_1=2.,
            tau_2=0.25,
            tau_3=0.5,
            device='cpu',
            num_restarts=1,
            **kwargs
    ):
        self._tau_0 = tau_0
        self._tau_1 = tau_1
        self._tau_2 = tau_2
        self._tau_3 = tau_3
        self._uncessessfull_trials = 0.
        self._num_restarts = max(1, int(num_restarts))
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
        used_samples = 0
        X = []
        y = []
        success = None
        rho = None
        for _ in range(self._num_restarts):
            used_samples += 2
            optimizer.update(oracle=model, x=previous_psi, step=step)
            current_psi, status, history = optimizer.optimize()

            func_prev_surr = model.func(previous_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
            func_curr_surr = model.func(current_psi, num_repetitions=num_repetitions).detach().clone().to(self._device)
            # y_model.generate()
            func_prev, X_prev = y_model.generate_data_at_point(n_samples_per_dim=num_repetitions, current_psi=previous_psi)
            func_curr, X_curr = y_model.generate_data_at_point(n_samples_per_dim=num_repetitions, current_psi=current_psi)
            X.append(X_prev); y.append(func_prev); X.append(X_curr); y.append(func_curr)

            std = (
                    (
                            func_prev.std() / (len(func_prev) - 1)**(0.5) +
                            func_prev.std() / (len(func_prev) - 1)**(0.5)
                    )
            ).item()
            # correction to garantee convergance to zero of step
            std = std / (1 + self._uncessessfull_trials)
            func_prev = func_prev.mean()
            func_curr = func_curr.mean()
            rho = ((func_prev - func_curr) / (func_prev_surr - func_curr_surr)).item()
            success = False
            # TODO: check if std should be corrected by _uncessessfull_trials
            if rho > self._tau_0 and (func_prev - func_curr).item() >= std:
                psi = current_psi
                success = True
                self._uncessessfull_trials = max(0, self._uncessessfull_trials - 1)
                if rho < self._tau_2 or (current_psi - previous_psi).norm().item() < 1e-2 * step:
                    optimizer.reverse_optimizer()

            else:
                psi = previous_psi
                # self._uncessessfull_trials += 1
                optimizer.reverse_optimizer()
            if not success:
                # if change is marginall we are willing to give it a chance
                if (func_prev - func_curr).abs().item() >= 2 * std:
                    break
            else:
                break

        if not success:
            self._uncessessfull_trials += 1

        step_status = None
        # TODO: add condition on if step is reliable
        # i.e. if \delta \psi > trust_region => increase
        # i.e. if model generalizes well => increase trust region
        # https://arxiv.org/pdf/1807.07994.pdf
        # if success and rho > 0.25 => increase step
        if rho > self._tau_2 and success:
            step_status = "Full success"
            min_ = step
            max_ = step * self._tau_1
            if used_samples > 2:
                max_ = step
            step = np.random.uniform(low=min_, high=max_)
        # if rho < 0.25 => decrease step
        elif rho < self._tau_2:
            step_status = "rho is small"
            # TODO: if alpha = 0.5, std >> step then we are too fast reduce step size
            alpha = 0.9
            _min = min(self._tau_3 * alpha * step + (1 - alpha) * std, step)
            _max = max(_min, step)
            step = np.random.uniform(
                low=_min,
                high=_max
            )
        # if rho > 0.25 but not success?
        # Meaning that std is high in comp with \Delta func
        # Or lfbfgs have wrong direction in mind
        else:
            # if std is small => increase step
            if (func_prev - func_curr).abs().item() <= std and (current_psi - previous_psi).norm().item() > step * 1e-2:
                step_status = "Std is small"
                min_ = step
                max_ = step * self._tau_1
                step = np.random.uniform(low=min_, high=max_)
            else:
                step_status = "std is ok, rho > 0.25, so wrong direction?"

        return psi, step, optimizer, {
            'diff_real': (func_prev - func_curr),
            'diff_surrogate': (func_prev_surr - func_curr_surr),
            'success': success,
            'proposed_psi': current_psi.detach().cpu().numpy(),
            'grad': -model.grad(previous_psi, num_repetitions=num_repetitions).detach().cpu().numpy(),
            'grad_corrected': (current_psi - previous_psi).detach().cpu().numpy(),
            'step_status': step_status,
            'X': torch.cat(X, dim=0),
            'y': torch.cat(y, dim=0),
            'used_samples': used_samples
        }


class TrustRegionSymmetric:
    def __init__(
            self,
            tau_0=1e-3,
            tau_1=2.,
            tau_2=0.25,
            tau_3=0.5,
            device='cpu',
            num_restarts=1,
            **kwargs
    ):
        self._tau_0 = tau_0
        self._tau_1 = tau_1
        self._tau_2 = tau_2
        self._tau_3 = tau_3
        self._uncessessfull_trials = 0.
        self._num_restarts = max(1, int(num_restarts))
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
        used_samples = 0
        X = []
        y = []
        success = None
        rho = None
        std = None

        for _ in range(self._num_restarts):
            used_samples += 2
            optimizer.update(oracle=model, x=previous_psi, step=step)
            current_psi, status, history = optimizer.optimize()

            func_prev_surr = model.func(previous_psi, num_repetitions=int((1 + self._uncessessfull_trials) * num_repetitions)).detach().clone().to(self._device)
            func_curr_surr = model.func(current_psi, num_repetitions=int((1 + self._uncessessfull_trials) * num_repetitions)).detach().clone().to(self._device)
            # y_model.generate()
            func_prev, X_prev = y_model.generate_data_at_point(n_samples_per_dim=num_repetitions, current_psi=previous_psi)
            func_curr, X_curr = y_model.generate_data_at_point(n_samples_per_dim=num_repetitions, current_psi=current_psi)
            X.append(X_prev); y.append(func_prev); X.append(X_curr); y.append(func_curr)

            std = (
                    (
                            func_prev.std() / (len(func_prev) - 1)**(0.5) +
                            func_prev.std() / (len(func_prev) - 1)**(0.5)
                    )
            ).item()
            # correction to garantee convergance to zero of step
            std = std / (1 + self._uncessessfull_trials)
            func_prev = func_prev.mean()
            func_curr = func_curr.mean()
            rho = ((func_prev - func_curr) / (func_prev_surr - func_curr_surr)).item()
            success = False
            # TODO: check if std should be corrected by _uncessessfull_trials
            if rho > self._tau_0 and (func_prev - func_curr).item() >= std:
                psi = current_psi
                success = True
                self._uncessessfull_trials = max(0, self._uncessessfull_trials - 1)
                if rho < self._tau_2 or (current_psi - previous_psi).norm().item() < 1e-2 * step:
                    optimizer.reverse_optimizer()
            else:
                psi = previous_psi
                # self._uncessessfull_trials += 1
                optimizer.reverse_optimizer()
            if not success:
                # if change is marginall we are willing to give another a chance
                if (func_prev - func_curr).abs().item() >= 2 * std:
                    break
            else:
                break

        if not success:
            self._uncessessfull_trials += 1

        step_status = None
        if rho > self._tau_2:
            if success:
                step_status = "Full success"
                min_ = step
                max_ = step * self._tau_1
                step = np.random.uniform(low=min_, high=max_)
            else:
                if (func_prev - func_curr).abs().item() <= std and \
                        (current_psi - previous_psi).norm().item() > step * 1e-2:
                    step_status = "Std is small"
                    min_ = step
                    max_ = step * self._tau_1
                    step = np.random.uniform(low=min_, high=max_)
                else:
                    pass
        else:
            if success:
                pass
            else:
                min_ = step * self._tau_3
                max_ = step
                step = np.random.uniform(low=min_, high=max_)

        # step = np.clip(step, std, np.inf)
        return psi, step, optimizer, {
            'diff_real': (func_prev - func_curr),
            'diff_surrogate': (func_prev_surr - func_curr_surr),
            'success': success,
            'proposed_psi': current_psi.detach().cpu().numpy(),
            'grad': -model.grad(previous_psi, num_repetitions=num_repetitions).detach().cpu().numpy(),
            'grad_corrected': (current_psi - previous_psi).detach().cpu().numpy(),
            'step_status': step_status,
            'X': torch.cat(X, dim=0),
            'y': torch.cat(y, dim=0),
            'used_samples': used_samples
        }
