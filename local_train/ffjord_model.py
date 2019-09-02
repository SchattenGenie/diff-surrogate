import torch
from base_model import BaseConditionalGenerationOracle
import sys
sys.path.append('./ffjord/')
import ffjord
import ffjord.lib
import ffjord.lib.utils as utils
from ffjord.lib.visualize_flow import visualize_transform
import ffjord.lib.layers.odefunc as odefunc
from ffjord.train_misc import standard_normal_logprob
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from ffjord.custom_model import build_model_tabular, get_transforms, compute_loss
import lib.layers as layers
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class FFJORDModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 x_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 num_blocks: int = 3,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 hidden_dims: Tuple[int] = (32, 32),
                 **kwargs):
        super(FFJORDModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        self._x_dim = x_dim
        self._y_dim =y_dim
        self._psi_dim = psi_dim

        self._model = build_model_tabular(dims=self._y_dim,
                                          condition_dim=self._psi_dim + self._x_dim,
                                          layer_type='concat_v2',
                                          num_blocks=num_blocks,
                                          rademacher=False,
                                          nonlinearity='tanh',
                                          solver='rk4',
                                          hidden_dims=hidden_dims,
                                          bn_lag=0.01,
                                          batch_norm=True,
                                          regularization_fns=None)
        self._sample_fn, self._density_fn = get_transforms(self._model)
        self._epochs = epochs
        self._lr = lr

    def loss(self, y, condition):
        return compute_loss(self._model, data=y.detach(), condition=condition.detach())

    def fit(self, y, condition):
        trainable_parameters = list(self._model.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=self._lr)
        for epoch in range(self._epochs):
            loss = self.loss(y, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def generate(self, condition):
        n = len(condition)
        z = torch.randn(n, self._y_dim).to(self.device)
        return self._sample_fn(z, condition)

    def log_density(self, y, condition):
        return self._density_fn(y, condition)

    def train(self):
        super().train()
        for module in self._model.modules():
            if hasattr(module, 'odeint'):
                module.__setattr__('odeint', odeint_adjoint)

    def eval(self):
        super().train()
        for module in self._model.modules():
            if hasattr(module, 'odeint'):
                module.__setattr__('odeint', odeint)
