import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as dist
import numpy as np
from itertools import product


class AdaptiveBorders:
    def __init__(self, psi_dim, step, device='cpu', lr=1e-1):
        self._psi_dim = psi_dim
        s = np.log(np.exp(step) - 1.)
        self._device = device
        self._log_sigma = torch.tensor(psi_dim * [s], dtype=torch.float32, device=device, requires_grad=True)
        self._optimizer = torch.optim.Adam([self._log_sigma], lr=lr)

    def step(self, model, conditions_grid, r_grid, num_repetitions=10000):
        """
        NotaBene: assumed that conditions_grid[-1] is a central point(where we estimate gradients)
        :param model:
        :param conditions_grid:
        :return:
        """
        self._optimizer.zero_grad()
        grad_surrogate = model.grad(conditions_grid, num_repetitions=num_repetitions).detach().clone().to(self._device)
        idx_1, idx_2 = list(map(list, list(zip(*list(product(
            range(len(conditions_grid))[:-1], [len(conditions_grid) - 1]
        ))))))
        grad_var = grad_surrogate[idx_2]
        condition_var = (conditions_grid[idx_2] - conditions_grid[idx_1]).to(self._device)
        r_var = (r_grid[idx_2] - r_grid[idx_1]).to(self._device)
        r = (r_var / (grad_var * condition_var).sum(dim=1)).abs().log().abs().view(-1, 1)
        loss = (
                r * dist.Normal(
            loc=torch.zeros_like(self.sigma),
            scale=self.sigma
        ).log_prob(condition_var)).mean()
        loss.backward()
        self._optimizer.step()

    @property
    def sigma(self):
        return F.softplus(self._log_sigma)

    @property
    def log_sigma(self):
        return self._log_sigma

    def log(self, experiment):
        sigma = self.sigma.detach().clone().numpy()
        if not (self.sigma.grad is None):
            sigma_grad = self.sigma.grad.detach().clone().numpy()
        else:
            sigma_grad = np.zeros(len(sigma))
        for i in range(len(self._log_sigma)):
            experiment.log_metric('sigma_{}'.format(i), sigma[i])
            experiment.log_metric('sigma_grad_{}'.format(i), sigma_grad[i])