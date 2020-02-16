import torch
import pyro
import numpy as np
from pyro import distributions as dist
from local_train.base_model import BaseConditionalGenerationOracle
from sklearn.datasets import load_boston
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import norm
from pyro import poutine
import matplotlib.pyplot as plt
import scipy
from pyDOE import lhs
import seaborn as sns
import lhsmdu
import tqdm
import requests
import traceback
import json
import time
import os


from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def average_block_wise(x, num_repetitions):
    n = x.shape[0]
    if len(x.shape) == 1:
        return F.avg_pool1d(x.view(1, 1, n),
                            kernel_size=num_repetitions,
                            stride=num_repetitions).view(-1)
    elif len(x.shape) == 2:
        cols = x.shape[1]
        return F.avg_pool1d(x.view(1, cols, n),
                            kernel_size=num_repetitions,
                            stride=num_repetitions).view(-1, cols)
    else:
        NotImplementedError("average_block_wise do not support >2D tensors")


class YModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10), y_dim=1,
                 loss=lambda y, **kwargs: OptLoss.SigmoidLoss(y, 5, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=y_dim)  # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
        self._y_dim = y_dim

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
        return pyro.sample('x', self._x_dist, torch.Size([sample_size])).to(self.device).view(-1, self._x_dim)

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

    def generate_data_at_point(self, n_samples_per_dim, current_psi):
        xs = self.sample_x(n_samples_per_dim)
        mus = current_psi.repeat(n_samples_per_dim, 1).clone().detach()
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)

    def generate_local_data(self, n_samples_per_dim, step, current_psi, std=0.1):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))
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
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))

        if n_samples == 0:
            mus = torch.zeros(0, len(current_psi)).float().to(self.device)
        else:
            mus = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)
        mus = step * (mus * 2 - 1) + current_psi
        mus = torch.cat([mus, current_psi.view(1, -1)])
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs_normal(self, n_samples_per_dim, sigma, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))
        mus = np.append(lhs(len(current_psi), n_samples), np.ones((1, len(current_psi))) / 2., axis=0)
        for i in range(len(current_psi)):
            mus[:, i] = norm(loc=current_psi[i].item(), scale=sigma[i].item()).ppf(
                mus[:, i]
            )
        mus = torch.tensor(mus).float().to(self.device)
        conditions_grid = mus.clone().detach()
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        r_grid = average_block_wise(self.loss(data), n_samples_per_dim)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1), conditions_grid, r_grid


class RosenbrockModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, *args, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class RosenbrockModelNoisless(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)

    def sample_x(self, sample_size):
        return pyro.sample('x', self._x_dist, torch.Size([sample_size])).to(self.device).view(-1, self._x_dim)

    def _generate_dist(self, psi, x):
        latent_psi = self.g(psi)
        return dist.Delta(latent_psi)


class Hartmann6(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-0.01, 0.01),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1)  # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
        self.alpha = torch.tensor([1.00, 1.20, 3.00, 3.20]).float().to(device)
        self.A = torch.tensor(
            [[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
             [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
             [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
             [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]]
        ).float().to(device)
        self.P = 0.0001 * torch.tensor(
            [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]]
        ).float().to(device)

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        latent_psi = self.g(psi)
        return dist.Normal(latent_psi, self.std_val(latent_x))

    def g(self, x):
        external_sum = torch.zeros(len(x)).float().to(x)
        for i in range(4):
            internal_sum = torch.zeros(len(x)).float().to(x)
            for j in range(6):
                internal_sum = internal_sum + self.A[i, j] * (x[:, j] - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * torch.exp(-internal_sum)

        return -external_sum.view(-1, 1)


def generate_covariance(n=100, a=2):
    np.random.seed(1337)
    A = np.matrix([np.random.randn(n) + np.random.randn(1) * a for i in range(n)])
    A = A * np.transpose(A)
    D_half = np.diag(np.diag(A)**(-0.5))
    C = D_half * A * D_half
    return np.array(C)


def generate_orthogonal_matrix_embedder(in_dim, out_dim, seed=1337):
    assert in_dim > out_dim
    mixing_covar, _ = np.linalg.qr(np.random.randn(in_dim, out_dim))
    return mixing_covar


def init_orthogonal_embedder(psi_dim, out_dim, seed=1337):
    deep_embedder = nn.Sequential(
        nn.Linear(psi_dim, 16, bias=False),
        nn.Tanh(),
        nn.Linear(16, out_dim, bias=False),
    )
    ortho_matrix_1 = generate_orthogonal_matrix_embedder(psi_dim, 16, seed=seed + 1)
    lin_1 = list(deep_embedder[0].parameters())[0]
    lin_1.data = torch.tensor(ortho_matrix_1.T).to(lin_1)
    ortho_matrix_2 = generate_orthogonal_matrix_embedder(16, out_dim, seed=seed + 2)
    lin_2 = list(deep_embedder[-1].parameters())[0]
    lin_2.data = torch.tensor(ortho_matrix_2.T).to(lin_2)
    return deep_embedder


class GaussianMixtureHumpModelDegenerate(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=1,
                 loss = lambda y, **kwargs: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
        torch.manual_seed(1337)
        np.random.seed(1337)
        assert self._psi_dim > 2
        mixing_covar, _ = np.linalg.qr(np.random.randn(self._psi_dim, 2))
        np.random.seed()
        torch.manual_seed(np.random.get_state()[1][-1])
        self._mixing_covariance = torch.tensor(mixing_covar).float().to(self._device)

    def _generate_dist(self, psi, x):
        return self.mixture_model(psi, x)

    def _generate(self, psi, x):
        psi = torch.mm(psi, self._mixing_covariance)
        return pyro.sample('y', self._generate_dist(psi, x))

    def mixture_model(self, psi, x, K=2):
        locs = pyro.sample('locs', dist.Normal(x * self.three_hump(psi).view(-1, 1), 1.)).to(self.device)
        #scales = pyro.sample('scale', dist.LogNormal(0., 2), torch.Size([len(x)])).view(-1, 1).to(self.device)
        assignment = pyro.sample('assignment', dist.Categorical(torch.abs(psi)))
        return dist.Normal(locs.gather(1, assignment.unsqueeze(1)), 1)

    # Three hump function http://benchmarkfcns.xyz/2-dimensional
    def three_hump(self, y):
        return 2 * y[:, 0] ** 2 - 1.05 * y[:, 0] ** 4 + y[:, 0] ** 6 / 6 + y[:,0] * y[:,1] + y[:, 1] ** 2


class GaussianMixtureHumpModelDeepDegenerate(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=1,
                 loss=lambda y, *args, **kwargs: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        torch.manual_seed(1337)
        self._deep_embedder = init_orthogonal_embedder(self._psi_dim, 2).to(device)
        torch.manual_seed(np.random.get_state()[1][-1])
        self._device = device
        self.loss = loss

    def _generate_dist(self, psi, x):
        return self.mixture_model(psi, x)

    def _generate(self, psi, x):
        deep_psi = self._deep_embedder(psi)
        return pyro.sample('y', self._generate_dist(deep_psi, x))

    def mixture_model(self, psi, x, K=2):
        locs = pyro.sample('locs', dist.Normal(x * self.three_hump(psi).view(-1, 1), 1.)).to(self.device)
        #scales = pyro.sample('scale', dist.LogNormal(0., 2), torch.Size([len(x)])).view(-1, 1).to(self.device)
        assignment = pyro.sample('assignment', dist.Categorical(torch.abs(psi)))
        return dist.Normal(locs.gather(1, assignment.unsqueeze(1)), 1)

    # Three hump function http://benchmarkfcns.xyz/2-dimensional
    def three_hump(self, y):
        return 2 * y[:, 0] ** 2 - 1.05 * y[:, 0] ** 4 + y[:, 0] ** 6 / 6 + y[:,0] * y[:,1] + y[:, 1] ** 2


class RosenbrockModelDeepDegenerate(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(1337)
        self._deep_embedder = init_orthogonal_embedder(self._psi_dim, 10).to(device)
        torch.manual_seed(np.random.get_state()[1][-1])
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        deep_psi = self._deep_embedder(psi)
        latent_psi = self.g(deep_psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class RosenbrockModelDegenerate(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, *args, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1)  # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(1337)
        np.random.seed(1337)
        mixing_covar, _ = np.linalg.qr(np.random.randn(self._psi_dim, 10))
        np.random.seed()
        self._mixing_covariance = torch.tensor(mixing_covar).float().to(self._device)
        torch.manual_seed(np.random.get_state()[1][-1])
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        psi = torch.mm(psi, self._mixing_covariance)
        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1,
                                                                                                   keepdim=True)


class ModelDegenerate(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(1337)
        # self._mixing_matrix = torch.tensor(scipy.linalg.hilbert(self._psi_dim)[:, :2]).float().to(self._device)
        self._mixing_matrix = torch.randn(self._psi_dim, 500).float().to(self._device)
        # self._mixing_covariance = torch.randn(self._psi_dim, self._psi_dim).float().to(self._device)
        self._mixing_covariance = torch.tensor(np.linalg.cholesky(generate_covariance(n=self._psi_dim))).float().to(self._device)
        torch.seed()
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)

        psi_z = dist.Normal(torch.zeros_like(psi), torch.ones_like(psi) / 100.).sample()
        psi = torch.mm(psi_z, self._mixing_covariance) + psi
        # psi = torch.mm(psi, self._mixing_matrix)
        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))


class RosenbrockModelInstrict(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(1337)
        self._mask = (torch.range(0, self._psi_dim - 1) % 2 == 0).byte()
        torch.seed()
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        psi = psi[:, self._mask]
        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class ModelInstrict(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(1337)
        self._mask = (torch.range(0, self._psi_dim - 1) % 2 == 0).byte()
        torch.seed()
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        psi = psi[:, self._mask]
        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class RosenbrockModelDegenerateInstrict(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1)  # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(1337)
        self._mask = (torch.range(0, self._psi_dim - 1) % 10 == 0).byte()
        self._mixing_covariance = torch.tensor(np.linalg.cholesky(generate_covariance(n=self._mask.sum()))).float().to(
            self._device)
        torch.seed()
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        psi = psi[:, self._mask]
        # psi_z = dist.Normal(torch.zeros_like(psi), torch.ones_like(psi) / 100.).sample()
        psi = torch.mm(psi, self._mixing_covariance)

        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)



class MultimodalSingularityModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y, **kwargs: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    @staticmethod
    def g(x):
        return x.abs().sum(dim=1, keepdim=True) * ((-x.pow(2).sin().sum(dim=1, keepdim=True)).exp())


class GaussianMixtureHumpModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=1,
                 loss = lambda y, *args, **kwargs: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    def _generate_dist(self, psi, x):
        return self.mixture_model(psi, x)

    def _generate(self, psi, x):
        return pyro.sample('y', self._generate_dist(psi, x))

    def mixture_model(self, psi, x, K=2):
        locs = pyro.sample('locs', dist.Normal(x * self.three_hump(psi).view(-1, 1), 1.)).to(self.device)
        #scales = pyro.sample('scale', dist.LogNormal(0., 2), torch.Size([len(x)])).view(-1, 1).to(self.device)
        assignment = pyro.sample('assignment', dist.Categorical(torch.abs(psi)))
        return dist.Normal(locs.gather(1, assignment.unsqueeze(1)), 1)

    # Three hump function http://benchmarkfcns.xyz/2-dimensional
    def three_hump(self, y):
        return 2 * y[:, 0] ** 2 - 1.05 * y[:, 0] ** 4 + y[:, 0] ** 6 / 6 + y[:,0] * y[:,1] + y[:, 1] ** 2


class LearningToSimGaussianModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_dim=1, y_dim=3):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Delta(torch.Tensor([0]).to(device))
        self._psi_dim = len(psi_init)
        self._device = device
        torch.manual_seed(0)
        np.random.seed(0)
        self.create_test_data()
        self.train_discriminator()
        torch.seed()
        np.random.seed()
        print("INIT_DONE")

    def create_test_data(self):
        class_size = 250
        self.n_class_params = 12
        self.true_params = {0: {0: {"mean": [-7.5, 0], "var": [0.5, 1e-15]},
                                1: {"mean": [-3, 3.], "var": [1, 0.5]},
                                2: {"mean": [3, -3.], "var": [1, 0.5]}},
                            1: {0: {"mean": [0, 5], "var": [0.5, 1e-15]},
                                1: {"mean": [3, 3.], "var": [1, 0.5]},
                                2: {"mean": [-3, -3.], "var": [1, 0.5]}}}

        # this is just a messy way to induce readability of what parameter means what in the psi tensor
        # and double check everything works with one tensor
        data_generation = []
        n_components = 3
        for class_index in [1, 0]:
            for i in range(n_components):
                data_generation.extend([*self.true_params[class_index][i]["mean"],
                                        *self.true_params[class_index][i]["var"]])
        self.psi_true = torch.Tensor(data_generation).repeat(2 * class_size, 1).to(self._device)

        #psi_as_dict = {1: self.psi_true[:, :self.n_class_params], 0: self.psi_true[:, self.n_class_params:]}
        psi_as_dict = self.psi_true.reshape(-1, 2, 12).transpose(0, 1)
        self.test_data = self.sample_toy_data_pt(n_classes=2, n_components=3, psi=psi_as_dict).to(self._device)

    def train_discriminator(self):
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 10),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(10, 16),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(16, 1)).to(self._device)

        opt = torch.optim.Adam(self.net.parameters())

        n_epochs = 200
        for e in range(n_epochs):
            output = self.net(self.test_data[:, :-1])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, self.test_data[:, -1].reshape(-1,1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            # with torch.no_grad():
            #     output = net(train_data[:, :-1])
            #     loss = torch.nn.functional.cross_entropy(output, train_data[:, -1].long())
            #     val_metric = accuracy_score(train_data[:, -1].long(), output.argmax(dim=1))
            #     print(loss.item(), val_metric)

        for param in self.net.parameters():
            param.requires_grad_(False)
        print(loss.item())


    def loss(self, y):
        self.net.zero_grad()
        output = self.net(y[:, :-1])
        mask = y[:, -1] > 0.5
        regulariser = y[:, -1][mask].mean()
        lam = 1
        # y = y[:, -1].reshape(-1, 1)
        # c = 2.
        return torch.nn.functional.binary_cross_entropy_with_logits(output,
                                                                    torch.clamp(y[:, -1].reshape(-1,1), 0., 1.),
                                                                    reduction='none') + lam * (regulariser - 1) ** 2

    def sample_toy_data_pt(self, n_classes=2, n_components=3, psi=None):
        means_index = [0, 1, 4, 5, 8, 9]
        std_index = [2, 3, 6, 7, 10, 11]

        n_samples = len(psi[0])
        classes_mask = pyro.sample('class_selection',
                                   dist.Categorical(torch.Tensor([1 / 2, 1 / 2]).view(1, -1).repeat(n_samples, 1)))
        classes_mask = classes_mask.to(self._device)

        data = []
        for class_index in pyro.plate("y", n_classes):
            probs = torch.Tensor([1. / n_components] * n_components).repeat(n_samples, 1)
            assignment = pyro.sample('assignment', dist.Categorical(probs))#.to(self._device)
            means = psi[class_index][:, means_index].reshape(-1, 3, 2).to(torch.device('cpu'))
            stds = psi[class_index][:, std_index].reshape(-1, 3, 2).repeat(1, 1, 2).reshape(-1, 3, 2, 2).to(torch.device('cpu'))
            stds = torch.stack([stds[:, :, 0, :], stds[:, :, 1, [1, 0]]], dim=-2)
            # inplace operator sometimes crashes optimization procedure
            # stds[:, :, 1, :] = stds[:, :, 1, [1, 0]]
            n_dist = dist.MultivariateNormal(means.gather(1, assignment.view(-1, 1).unsqueeze(2).repeat(1, 1, 2)),
                                             stds.gather(1,  assignment.view(-1, 1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 2)))

            data.append(pyro.sample("y_{}".format(class_index), n_dist))
        data = torch.stack(data).to(self._device)
        data = data.gather(0, classes_mask.view(1, -1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 2))[0, :, 0, :]
        data = torch.cat([data, classes_mask.view(-1, 1).float()], dim=1)
        return data

    def _generate_dist(self, psi, x):
        #return self.mixture_model(psi, x)
        raise NotImplementedError

    def _generate(self, psi, x):
        #sigm_ind = list(sum([(i, i + 1) for i in range(2, 24, 4)], ()))
        #psi[:, sigm_ind] = torch.exp(psi[:, sigm_ind])

        # messy stuff to put fixed stds in place
        fixed_std_dim = list(sum([(i, i + 1) for i in range(2, 24, 4)], ()))
        mu_dim = list(sum([(i, i + 1) for i in range(0, 24, 4)], ()))

        output = torch.ones([len(psi), self.psi_true.shape[1]]).to(self._device)
        output[:, mu_dim] = psi
        output[:, fixed_std_dim] = self.psi_true[:1, fixed_std_dim].repeat(len(psi), 1)

        psi_as_dict = output.reshape(-1, 2, 12).transpose(0,1)
        return self.sample_toy_data_pt(psi=psi_as_dict)


class FreqModulatedSoundWave(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=2,
                 loss=lambda y, **kwargs: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class LennardJonesPotentialProblem(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=2,
                 loss = lambda y, **kwargs: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
    
    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.object):
            if obj.ndim == 0:
                return float(obj)
            elif obj.ndim == 1:
                return [float(t) for t in obj.tolist()]
        return json.JSONEncoder.default(self, obj)


class BernoulliModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=(-0.1, 0.1),
                 x_dim=1,
                 y_dim=1,
                 loss=lambda y, **kwargs: (y - 0.499).pow(2).mean(dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    def _generate_dist(self, psi, x):
        latent_psi = torch.sigmoid(psi + x)
        return dist.RelaxedBernoulli(torch.tensor(0.0001).float().to(psi.device), probs=latent_psi)

    def _generate(self, psi, x):
        return pyro.sample('y', self._generate_dist(psi, x))

    def generate(self, condition):
        psi, x = condition[:, :self._psi_dim], condition[:, self._psi_dim:]
        return self._generate(psi, x)

    def sample(self, sample_size):
        psi = self.sample_psi(sample_size)
        x = self.sample_x(sample_size)
        return self._generate(psi, x)


class BOCKModel(YModel):
    class BOCK_MNIST_net(torch.nn.Module):
        def __init__(self, N_hidden):
            super().__init__()
            self.N_hidden = N_hidden

            self.model = torch.nn.Sequential(
                torch.nn.Linear(784, N_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(N_hidden, 10))
            self.turn_off_w2_update()
            self.print_params_with_grad()

        def turn_off_w2_update(self):
            for name, param in self.model.named_parameters():
                if name == "2.weight":
                    param.requires_grad_(False)
                    print(param.shape)

        def set_param(self, param_value, param_name="2.weight"):
            for name, param in self.model.named_parameters():
                if name == param_name:
                    param.data = param_value.reshape(10, -1)

        def print_params_with_grad(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)

        def forward(self, X):
            return self.model(X)

    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_dim=784, y_dim=10):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Delta(torch.Tensor([0]).to(device))
        self._psi_dim = len(psi_init)
        self._device = device

        self.batch_size = 512
        self.epochs = 50
        self.create_data()

        bock_net_size = 10
        self.net = self.BOCK_MNIST_net(bock_net_size).to(device)
        self.net.set_param(psi_init)

        print("INIT_DONE")

    def net_loss(self, y_pred, y_true):
        return torch.nn.functional.cross_entropy(y_pred, y_true)

    def create_data(self, use_cache=True, normalise=False, data_path=os.path.expanduser("~/data/sklearn_datasets")):
        if use_cache and os.path.isfile(os.path.join(data_path, "mnist_x.npy")):
            X = np.load(os.path.join(data_path, "mnist_x.npy"), allow_pickle=True)
            y = np.load(os.path.join(data_path, "mnist_y.npy"), allow_pickle=True)
        else:
            X, y = fetch_openml('mnist_784', data_home="/mnt/shirobokov/sklearn_datasets", return_X_y=True, cache=True)
            np.save(os.path.join(data_path,"mnist_x.npy"), X)
            np.save(os.path.join(data_path,"mnist_y.npy"), y)

        X_train = X[:60000].astype(float)
        X_test = X[60000:].astype(float)

        self.y_train = torch.tensor(y[:60000].astype(int), dtype=torch.long)
        self.y_test = torch.tensor(y[60000:].astype(int), dtype=torch.long)
        if normalise:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        train_dataset = torch.utils.data.TensorDataset(self.y_train.to(self._device),
                                                       torch.tensor(X_train, dtype=torch.float).to(self._device))
        self.train_batch_gen = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True)

        test_dataset = torch.utils.data.TensorDataset(self.y_test.to(self._device),
                                                      torch.tensor(X_test, dtype=torch.float).to(self._device))
        self.test_batch_gen = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False)
        self.X_test = torch.tensor(X_test, dtype=torch.float).to(self._device)

    def train_net(self):
        optimizer = torch.optim.Adam(self.net.parameters())
        for epoch in trange(self.epochs):
            epoch_loss = 0
            for y_batch, X_batch in self.train_batch_gen:
                y_pred = self.net(X_batch)
                loss = self.net_loss(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        print("Train Loss in the end: {}".format(epoch_loss))

        y_score = []
        with torch.no_grad():
            for y_batch, X_batch in self.test_batch_gen:
                logits = self.net(X_batch)
                y_pred = logits.detach().cpu()
                y_score.append(y_pred)
        print("Test Loss in the end: {}".format(self.net_loss(torch.cat(y_score), torch.tensor(self.y_test))))

    def loss(self, y):
        return self.net_loss(y, self.y_test.to(y.device))

    def _generate_dist(self, psi, x):
        raise NotImplementedError

    def _generate(self, psi, x):
        # TODO: ITS not yet implemented in case we have a lot of different psi vectors in the input

        self.net.set_param(psi[:1, :])
        self.train_net()

        y_score = []
        with torch.no_grad():
            for y_batch, X_batch in self.test_batch_gen:
                logits = self.net(X_batch)
                y_pred = logits.detach().cpu()
                y_score.append(y_pred)
        return torch.cat(y_score)

    def sample_x(self, sample_size):
        return self.X_test.repeat((sample_size // len(self.X_test), 1))

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))

        if n_samples == 0:
            mus = torch.zeros(0, len(current_psi)).float().to(self.device)
        else:
            mus = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)
        mus = step * (mus * 2 - 1) + current_psi
        mus = torch.cat([mus, current_psi.view(1, -1)])
        data_list = []
        for mu in mus:
            data_list.append(self._generate(mu.reshape(1, -1), None))
        data = torch.cat(data_list).to(self.device)
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        return data, torch.cat([mus, xs], dim=1)


class BostonNNTuning(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1)  # hardcoded values
        assert len(psi_init) == 91
        self._psi_dim = 91
        self._device = device
        boston = load_boston()
        X, y = (boston.data, boston.target)
        self._X = torch.tensor(X).float().to(self._device)
        self._y = torch.tensor(y).float().view(-1, 1).to(self._device)
        torch.manual_seed(1337)
        self._net = nn.Sequential(
            nn.Linear(13, 6),
            nn.Tanh(),
            nn.Linear(6, 1)
        ).to(self._device)
        self._net.requires_grad_(False)
        self._d = dict([
            (tuple(x.detach().cpu().numpy().astype(np.float32)), y_.detach().cpu().numpy().astype(np.float32)[0])
            for x, y_ in zip(self._X, self._y)])
        torch.manual_seed(np.random.get_state()[1][-1])

    def _set_parameters(self, psi):
        self._net[0].weight.data = psi[: 6 * 13].view(6, 13).detach().clone().float().to(self._device)
        self._net[0].bias.data = psi[6 * 13: 6 * 13 + 6].view(6).detach().clone().float().to(self._device)

        self._net[2].weight.data = psi[6 * 13 + 6: 6 * 13 + 6 + 6].view(1, 6).detach().clone().float().to(self._device)
        self._net[2].bias.data = psi[6 * 13 + 6 + 6: 6 * 13 + 6 + 6 + 1].view(1).detach().clone().float().to(
            self._device)

    def sample_x(self, num_repetitions):
        sample_indices = np.random.choice(
            range(len(self._X)),
            num_repetitions,
            replace=True
        )
        return self._X[sample_indices]

    def _generate(self, psi, x):
        time.sleep(0.0000001)
        self._set_parameters(psi)
        return self._net(x)

    def generate(self, condition):
        if len(condition.view(-1)) > self._psi_dim + 13:
            RuntimeWarning('len(psi) > 91 is not supported in generate')
        psi = condition.view(-1)[:self._psi_dim]
        _, x = condition[:, :self._psi_dim], condition[:, self._psi_dim:]
        return self._generate(psi, x)

    def loss(self, y, conditions):
        psi, x = conditions[:, :self._psi_dim], conditions[:, self._psi_dim:]
        y_true = [self._d[tuple(x_.detach().cpu().numpy().astype(np.float32))] for x_ in x]
        y_true = torch.tensor(y_true).view(-1, 1).to(y)
        return (y - y_true).pow(2)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))
        if n_samples == 0:
            mus = torch.zeros(0, len(current_psi)).float().to(self.device)
        else:
            mus = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)
        mus = step * (mus * 2 - 1) + current_psi
        mus = torch.cat([mus, current_psi.view(1, -1)])
        data_list = []
        for i, mu in enumerate(mus):
            data_list.append(self._generate(mu, xs[i * n_samples_per_dim: (i + 1) * n_samples_per_dim]))
        data = torch.cat(data_list).to(self.device)
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        return data, torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs_normal(self, n_samples_per_dim, sigma, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))
        mus = np.append(lhs(len(current_psi), n_samples), np.ones((1, len(current_psi))) / 2., axis=0)
        for i in range(len(current_psi)):
            mus[:, i] = norm(loc=current_psi[i].item(), scale=sigma[i].item()).ppf(
                mus[:, i]
            )
        mus = torch.tensor(mus).float().to(self.device)
        data_list = []
        for i, mu in enumerate(mus):
            data_list.append(self._generate(mu, xs[i * n_samples_per_dim: (i + 1) * n_samples_per_dim]))
        data = torch.cat(data_list).to(self.device)
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1), None, None

    def generate_data_at_point(self, n_samples_per_dim, current_psi):
        xs = self.sample_x(n_samples_per_dim)
        data = self._generate(current_psi, xs)
        mus = current_psi.repeat(n_samples_per_dim, 1).clone().detach()
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)
