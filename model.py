import torch
import pyro
import numpy as np
from pyro import distributions as dist
import local_train.base_model as base_model
from pyro import poutine
import matplotlib.pyplot as plt
import seaborn as sns
import lhsmdu
import tqdm


class YModel(base_model.BaseConditionalGenerationOracle):
    def __init__(self, x_range=(-10, 10),
                 init_mu=torch.tensor(0.),
                 device='cpu'):
        self.mu_dist = dist.Delta(init_mu)
        self.x_dist = dist.Uniform(*x_range)
        self.condition_sample = None
        self._device = device
        # self.x_dist = dist.Delta(torch.tensor(float(0)))

    @property
    def device(self):
        return self._device

    @staticmethod
    def f(x, a=0, b=1, c=2):
        return a + b * x

    @staticmethod
    def g(x, d=2):
        # return -7 + x ** 2 / 10 + x ** 3 / 100
        # return d * torch.sin(x)
        return torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        # return x

    @staticmethod
    def std_val(x):
        return 0.1 + torch.abs(x) * 0.5

    def sample(self, sample_size=1):
        mu = pyro.sample('mu', self.mu_dist, torch.Size([sample_size]))
        size = [len(mu)]
        x = pyro.sample('x', self.x_dist, torch.Size(size))
        latent_x = pyro.sample('latent_x', dist.Normal(x, 1))
        latent_x = self.f(latent_x)
        latent_mu = self.g(mu)
        return pyro.sample('y', dist.Normal(latent_x + latent_mu, self.std_val(latent_x)))

    def generate(self, condition):
        mu, x = condition[:, :2], condition[:, 2:]

        latent_x = pyro.sample('latent_x', dist.Normal(x, 1))
        latent_x = self.f(latent_x)

        latent_mu = self.g(mu)
        return pyro.sample('y', dist.Normal(latent_x + latent_mu, self.std_val(latent_x)))

    def loss(self, x, condition):
        pass

    def fit(self, x, condition):
        pass

    def log_density(self, x, condition):
        pass

    def make_condition_sample(self, data):
        self.condition_sample = poutine.condition(self.sample, data=data)

    def generate_local_data(self, n_samples_per_dim, step, current_psi, x_dim=1, std=0.1):
        xs = self.x_dist.sample(
            torch.Size([n_samples_per_dim * 2 * current_psi.shape[1] + n_samples_per_dim, x_dim]))  # .to(device)

        mus = torch.empty((xs.shape[0], current_psi.shape[1]))  # .to(device)

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

        self.make_condition_sample({'mu': mus, 'X': xs})
        data = self.condition_sample().detach().to(device)
        return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, x_dim=1, n_samples=2):
        xs = self.x_dist.sample(torch.Size([
            n_samples_per_dim * n_samples,
            x_dim]))

        mus = torch.tensor(lhsmdu.sample(len(current_psi), n_samples,
                                         randomSeed=np.random.randint(1e5)).T).float().to(current_psi.device)

        mus = step * (mus * 2 - 1) + current_psi
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample().detach().to(current_psi.device)
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
