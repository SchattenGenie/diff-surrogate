import torch
from torch import nn
import torch.utils.data as dataset_utils
import copy
from base_model import BaseConditionalGenerationOracle
import sys
sys.path.append('./ffjord/')
import ffjord
import ffjord.lib
import ffjord.lib.utils as utils
from ffjord.lib.visualize_flow import visualize_transform
import ffjord.lib.layers.odefunc as odefunc
from ffjord.train_misc import standard_normal_logprob, create_regularization_fns
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from ffjord.custom_model import build_model_tabular, get_transforms, compute_loss
import lib.layers as layers
from tqdm import tqdm, trange
from typing import Tuple
import swats
import warnings
import numpy as np
import pyro
import pyro.distributions as dist
warnings.filterwarnings("ignore")


def logsumexp(x, dim):
    x_max, _ = x.max(dim=dim, keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res = x_max + torch.log((x - x_max_expand).exp().sum(dim=dim, keepdim=True))
    return res


class GaussianMixtureNetwork(nn.Module):
    def __init__(self, input_dim, mixture_size, targets, hidden_dim=32, device='cpu'):
        super(GaussianMixtureNetwork, self).__init__()
        self.input_dim = input_dim
        self.mixture_size = mixture_size
        self.targets = targets
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.softmax = nn.Softmax(dim=2)
        self.TWO = torch.tensor(2., dtype=torch.float32).to(device)
        self.ADDITIVE_TERM = torch.tensor(-0.5 * np.log(np.pi * 2), dtype=torch.float32).to(device)
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.alphas = nn.Linear(hidden_dim, mixture_size * targets)
        self.sigmas = nn.Linear(hidden_dim, mixture_size * targets)
        self.means = nn.Linear(hidden_dim, mixture_size * targets)

    def forward(self, inputs):
        hidden = self.nn(inputs)
        alphas = self.alphas(hidden).view(-1, self.targets, self.mixture_size)
        log_sigmas = self.sigmas(hidden).view(-1, self.targets, self.mixture_size)
        means = self.means(hidden).view(-1, self.targets, self.mixture_size)
        return alphas, means, torch.clamp_min(log_sigmas, -5)

    def logits(self, inputs, target):
        """
        inputs = [N, input_dim]
        target = [N, K]
        """
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)

        log_alphas = self.logsoftmax(alphas)
        log_pdf = self.ADDITIVE_TERM - log_sigmas - (((target.unsqueeze(-1) - means)) / log_sigmas.exp()).pow(
            2) / self.TWO
        logits = logsumexp(log_alphas + log_pdf, dim=-1).view(-1, self.targets)
        return logits

    def generate(self, inputs):
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)
        alphas = self.softmax(alphas)
        # alphas_picked = [N, K]
        alphas_sampled = pyro.sample("alphas", dist.Categorical(alphas))
        sigmas = log_sigmas.exp()
        result = pyro.sample("preds", dist.Normal(
            torch.gather(means.view(-1, self.mixture_size), dim=1, index=alphas_sampled.view(-1, 1)).view(-1,
                                                                                                          self.targets),
            torch.gather(sigmas.view(-1, self.mixture_size), dim=1, index=alphas_sampled.view(-1, 1)).view(-1,
                                                                                                           self.targets)
        ))
        return result

class GaussianMixtureNetwork(nn.Module):
    def __init__(self, input_dim, mixture_size, targets, hidden_dim=32, device='cpu'):
        super(GaussianMixtureNetwork, self).__init__()
        self.input_dim = input_dim
        self.mixture_size = mixture_size
        self.targets = targets
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.TWO = torch.tensor(2., dtype=torch.float32).to(device)
        self.ADDITIVE_TERM = torch.tensor(-0.5 * np.log(np.pi * 2), dtype=torch.float32).to(device)
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.alphas = nn.Linear(hidden_dim, mixture_size)
        self.sigmas = nn.Linear(hidden_dim, mixture_size * targets)
        self.means = nn.Linear(hidden_dim, mixture_size * targets)

    def forward(self, inputs):
        hidden = self.nn(inputs)
        alphas = self.alphas(hidden).view(-1, self.mixture_size)
        log_sigmas = self.sigmas(hidden).view(-1, self.targets, self.mixture_size)
        means = self.means(hidden).view(-1, self.targets, self.mixture_size)
        return alphas, means, torch.clamp_min(log_sigmas, -5) - 5

    def logits(self, inputs, target):
        """
        inputs = [N, input_dim]
        target = [N, K]
        """
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)

        log_alphas = self.logsoftmax(alphas)
        log_pdf = self.ADDITIVE_TERM - log_sigmas - (((target.unsqueeze(-1) - means)) / log_sigmas.exp()).pow(
            2) / self.TWO
        logits = logsumexp(log_alphas + log_pdf, dim=-1).view(-1, self.targets)
        return logits

    def generate(self, inputs):
        # alphas, means, sigmas = [N, K, mixture_size]
        alphas, means, log_sigmas = self.forward(inputs)
        alphas = self.softmax(alphas)
        # alphas_picked = [N, K]
        alphas_sampled = pyro.sample("alphas", dist.Categorical(alphas))
        # print(alphas_sampled)
        sigmas = log_sigmas.exp()

        result = pyro.sample("preds", dist.Normal(
            torch.gather(input=means, dim=2, index=alphas_sampled.view(-1, 1).repeat(1, self.targets).view(-1, self.targets, 1)).view(-1, self.targets),
            torch.gather(input=sigmas, dim=2, index=alphas_sampled.view(-1, 1).repeat(1, self.targets).view(-1, self.targets, 1)).view(-1, self.targets)
        ))
        return result

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, improvement=1e-4):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self._improvement = improvement
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self._improvement:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class GMMModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 x_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 num_blocks: int = 1,
                 lr: float = 1e-2,
                 epochs: int = 10,
                 mixture_size: int = 3,
                 **kwargs):
        super(GMMModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        self._x_dim = x_dim
        self._y_dim =y_dim
        self._psi_dim = psi_dim
        self._model = GaussianMixtureNetwork(input_dim=x_dim + psi_dim,
                                             mixture_size=mixture_size,
                                             targets=y_dim, device=y_model._device)
        self._epochs = epochs
        self._lr = lr

    def loss(self, y, condition, weights=None):
        return -self._model.logits(inputs=condition, target=y).mean(dim=1).mean()

    def fit(self, y, condition, weights=None):
        self.train()
        print(self.device)
        trainable_parameters = list(self._model.parameters())
        optimizer = swats.SWATS(trainable_parameters, lr=self._lr, verbose=True)
        best_params = self._model.state_dict()
        best_loss = 1e6
        early_stopping = EarlyStopping(patience=200, verbose=True)
        for epoch in range(self._epochs):
            optimizer.zero_grad()
            loss = self.loss(y + torch.randn_like(y) * y.std(dim=0) / 100, condition, weights=weights)
            print(loss)
            if loss.item() < best_loss:
                best_params = copy.deepcopy(self._model.state_dict())
                best_loss = loss.item()
            early_stopping(loss.item())
            if early_stopping.early_stop:
                break
            loss.backward()
            optimizer.step()
        self._model.load_state_dict(best_params)
        self.eval()
        self._sample_fn, self._density_fn = get_transforms(self._model)
        return self

    def generate(self, condition):
        return self._model.generate(condition)

    def log_density(self, y, condition):
        return self._model.logits(inputs=condition, target=y).mean(dim=1)

    def train(self):
        super().train(True)

    def eval(self):
        super().train(False)