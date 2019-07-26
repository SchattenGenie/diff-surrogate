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
from ffjord.custom_model import build_model_tabular, get_transforms
import lib.layers as layers
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class FFJORDModel(BaseConditionalGenerationOracle):
    def __init__(self, dim_x: int,
                 dim_condition: int,
                 num_blocks: int = 3,
                 hidden_dims: Tuple[int] = (32, 32)):
        super(FFJORDModel, self).__init__()
        self._model = build_model_tabular(dims=dim_x,
                                          condition_dim=dim_condition,
                                          layer_type='concat_v2',
                                          num_blocks=num_blocks,
                                          rademacher=False,
                                          nonlinearity='tanh',
                                          solver='rk4',
                                          hidden_dims=hidden_dims,
                                          bn_lag=0.01,
                                          batch_norm=True,
                                          regularization_fns=None)
        self._dim_x = dim_x
        self._dim_condition = dim_condition
        self._sample_fn, self._density_fn = get_transforms(self._model)

    def loss(self, x, condition):
        return compute_loss(self._model, data=x, condition=condition)

    def fit(self, x, condition, epochs=400, lr=1e-3):
        trainable_parameters = list(self._model.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

        for epoch in range(epochs):
            loss = self.loss(x, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def generate(self, condition):
        n = len(condition)
        z = torch.randn(n, self._dim_x).to(self.device)
        return self._sample_fn(z, condition)

    def log_density(self, x, condition):
        return self._density_fn(x, condition)
