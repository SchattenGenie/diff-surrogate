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
from ffjord.custom_model import build_model_tabular, get_transforms, compute_loss
import lib.layers as layers
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class FFJORDModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 x_dim: int,
                 condition_dim: int,
                 num_blocks: int = 3,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 hidden_dims: Tuple[int] = (32, 32)):
        super(FFJORDModel, self).__init__()
        self._model = build_model_tabular(dims=x_dim,
                                          condition_dim=condition_dim,
                                          layer_type='concat_v2',
                                          num_blocks=num_blocks,
                                          rademacher=False,
                                          nonlinearity='tanh',
                                          solver='rk4',
                                          hidden_dims=hidden_dims,
                                          bn_lag=0.01,
                                          batch_norm=True,
                                          regularization_fns=None)
        self._x_dim = x_dim
        self._condition_dim = condition_dim
        self._sample_fn, self._density_fn = get_transforms(self._model)
        self._epochs = epochs
        self._lr = lr

    def loss(self, x, condition):
        return compute_loss(self._model, data=x, condition=condition)

    def fit(self, x, condition):
        trainable_parameters = list(self._model.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=self._lr)

        for epoch in range(self._epochs):
            loss = self.loss(x, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def generate(self, condition):
        n = len(condition)
        z = torch.randn(n, self._x_dim).to(self.device)
        return self._sample_fn(z, condition)

    def log_density(self, x, condition):
        return self._density_fn(x, condition)
