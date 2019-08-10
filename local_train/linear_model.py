import torch
from base_model import BaseConditionalGenerationOracle
import sys


class LinearRegression(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class LinearModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 x_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 lr: float = 1e-3,
                 epochs: int = 10):
        super(LinearModel, self).__init__(y_model=y_model,
                                          x_dim=x_dim,
                                          psi_dim=psi_dim,
                                          y_dim=y_dim)
        self._x_dim = x_dim
        self._y_dim =y_dim
        self._psi_dim = psi_dim

        self._model = LinearRegression(self._psi_dim + self._x_dim, self._y_dim)
        self._epochs = epochs
        self._lr = lr
        self._loss = torch.nn.MSELoss(size_average = False)

    def loss(self, y, condition):
        y_predict = self._model(condition)
        return self._loss(y_predict, y)

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
        return self._model(condition)

    def log_density(self, y, condition):
        return None


class LinearModelOnPsi(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 x_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 lr: float = 1e-3,
                 epochs: int = 10):
        super(LinearModelOnPsi, self).__init__(y_model=y_model,
                                               x_dim=x_dim,
                                               psi_dim=psi_dim,
                                               y_dim=y_dim)
        self._x_dim = x_dim
        self._y_dim =y_dim
        self._psi_dim = psi_dim

        self._model = LinearRegression(self._psi_dim, self._y_dim)
        self._epochs = epochs
        self._lr = lr
        self._loss = torch.nn.MSELoss(size_average=False)

    def loss(self, y, condition):
        y_predict = self._model(condition)
        return self._loss(y_predict, y)

    def preprocess_dataset(self, y, condition):
        condition_psi = condition[:, :self._psi_dim]
        new_y, new_cond = [], []
        for cond in torch.unique(condition_psi, dim=0):
            mask = (condition_psi == cond).max(dim=1)[0]
            new_y.append(y[mask].mean())
            new_cond.append(condition_psi[mask].mean(dim=0))
        new_y = torch.stack(new_y)
        new_cond = torch.stack(new_cond)
        return new_y, new_cond

    def fit(self, y, condition):
        trainable_parameters = list(self._model.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=self._lr)
        y, condition = self.preprocess_dataset(y=y, condition=condition)
        for epoch in range(self._epochs):
            loss = self.loss(y, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def generate(self, condition):
        return self._model(condition[:, :self._psi_dim])

    def log_density(self, y, condition):
        return None
