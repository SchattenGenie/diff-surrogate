import torch
import pyro
import numpy as np
from pyro import distributions as dist
from local_train.base_model import BaseConditionalGenerationOracle
from pyro import poutine
import matplotlib.pyplot as plt
import seaborn as sns
import lhsmdu
import tqdm


class YModel(BaseConditionalGenerationOracle):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y: OptLoss.SigmoidLoss(y, 5, 10)):
        super(YModel, self).__init__(None)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)  # TODO: explicitly set psi_dim ?
        self._device = device
        self.loss = loss

    @property
    def _y_model(self):
        return self

    @property
    def device(self):
        return self._device

    @staticmethod
    def f(x, a=0, b=1):
        return a + b * x

    @staticmethod
    def g(x):
        return x.pow(2).sum(dim=1, keepdim=True).sqrt()

    @staticmethod
    def std_val(x):
        return 0.1 + x.abs() * 0.5

    def sample_psi(self, sample_size):
        return pyro.sample('mu', self._psi_dist, torch.Size([sample_size])).to(self.device)

    def sample_x(self, sample_size):
        return pyro.sample('x', self._x_dist, torch.Size([sample_size])).to(self.device).view(-1, 1)

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    def _generate(self, psi, x):
        return pyro.sample('y', self._generate_dist(psi, x))

    def generate(self, condition):
        psi, x = condition[:, :self._psi_dim], condition[:, self._psi_dim:]
        return self._generate(psi, x)

    def sample(self, sample_size):
        psi = self.sample_psi(sample_size)
        x = self.sample_x(sample_size)
        return self._generate(psi, x)

    def loss(self, y, condition):
        pass

    def fit(self, y, condition):
        pass

    def log_density(self, y, condition):
        psi, x = condition[:, :self._psi_dim], condition[:, self._psi_dim:]
        return self._generate_dist(psi, x).log_prob(y)

    def condition_sample(self):
        raise NotImplementedError("First call self.make_condition_sample")

    def make_condition_sample(self, data):
        self.condition_sample = poutine.condition(self.sample, data=data)

    def generate_local_data(self, n_samples_per_dim, step, current_psi, x_dim=1, std=0.1):
        xs = self.x_dist.sample(
            torch.Size([n_samples_per_dim * 2 * current_psi.shape[1] + n_samples_per_dim, x_dim])).to(self.device)

        mus = torch.empty((xs.shape[0], current_psi.shape[1])).to(self.device)

        iterator = 0
        for dim in range(current_psi.shape[1]):
            for dir_step in [-step, step]:
                random_mask = torch.torch.randn_like(current_psi)
                random_mask[0, dim] = 0
                new_psi = current_psi + random_mask * std
                new_psi[0, dim] += dir_step

                mus[iterator:
                    iterator + n_samples_per_dim, :] = new_psi.repeat(n_samples_per_dim, 1)
                iterator += n_samples_per_dim

        mus[iterator: iterator + n_samples_per_dim, :] = current_psi.repeat(n_samples_per_dim, 1).clone().detach()

        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample().detach().to(self.device)
        return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * n_samples)

        mus = torch.tensor(lhsmdu.sample(len(current_psi), n_samples,
                                         randomSeed=np.random.randint(1e5)).T).float().to(self.device)

        mus = step * (mus * 2 - 1) + current_psi
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)


class OptLoss(object):
    def __init__(self):
        pass
    
    @staticmethod
    def R(ys: torch.tensor, Y_0=-5):
        return (ys - Y_0).pow(2).mean(dim=1)

    @staticmethod
    def SigmoidLoss(ys: torch.tensor, left_bound, right_bound):
        return -torch.mean(torch.sigmoid(ys - left_bound) - torch.sigmoid(ys - right_bound), dim=1)

    @staticmethod
    def TanhLoss(ys: torch.tensor, left_bound, right_bound):
        return -torch.mean(torch.tanh(ys - left_bound) - torch.tanh(ys - right_bound), dim=1)
