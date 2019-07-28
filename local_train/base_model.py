from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from hessian import hessian as hessian_calc
import sys


class BaseConditionalGeneratorModel(nn.Module, ABC):
    """
    Base class for implementation of conditional generation model.
    In our case condition is concatenation of psi and x,
    i.e. condition = torch.cat([psi, x], dim=1)
    """

    @property
    def device(self):
        """
        Just a nice class to get current device of the model
        Might be error-prone thou
        :return: torch.device
        """
        return next(self.parameters()).device

    @abstractmethod
    def fit(self, y, condition):
        """
        Computes the value of function at point x.
        :param y: target variable
        :param condition: torch.Tensor
            Concatenation of [psi, x]
        """
        raise NotImplementedError('fit is not implemented.')

    @abstractmethod
    def loss(self, y, condition):
        """
        Computes model loss for given y and condition.
        """
        raise NotImplementedError('loss is not implemented.')

    @abstractmethod
    def generate(self, condition):
        """
        Generates samples for given conditions
        """
        raise NotImplementedError('predict is not implemented.')

    @abstractmethod
    def log_density(self, y, condition):
        """
        Computes log density for given conditions and y
        """
        raise NotImplementedError('log_density is not implemented.')


class BaseConditionalGenerationOracle(BaseConditionalGeneratorModel, ABC):
    """
    Base class for implementation of loss oracle.
    """
    def __init__(self, y_model):
        super(BaseConditionalGenerationOracle, self).__init__()
        self.__y_model = y_model

    @property
    def _y_model(self):
        return self.__y_model

    def func(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the value of function with specified condition.
        :param condition: torch.Tensor
            condition of models, i.e. psi
            # TODO: rename condition -> psi?
        :param num_repetitions:
        :return:
        """
        condition = condition.to(self.device)
        if isinstance(num_repetitions, int):
            assert len(condition.size()) == 1
            conditions = condition.repeat(num_repetitions, 1)
            conditions = torch.cat([
                conditions,
                self._y_model.sample_x(len(conditions)).to(self.device)
            ], dim=1)
            y = self.generate(conditions)
            loss = self._y_model.loss(y=y).mean()
            return loss
        else:
            condition = torch.cat([
                condition,
                self._y_model.sample_x(len(condition)).to(self.device)
            ], dim=1)
            y = self.generate(condition)
            loss = self._y_model.loss(y=y)
            return loss

    def grad(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the gradient of function with specified condition.
        If num_repetitions is not None then condition assumed
        to be 1-d tensor, which would be repeated num_repetitions times
        :param condition: torch.Tensor
            2D or 1D array on conditions for generator
        :param num_repetitions:
        :return: torch.Tensor
            1D torch tensor
        """
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        if isinstance(num_repetitions, int):
            assert len(condition.size()) == 1
            return grad([self.func(condition, num_repetitions=num_repetitions)], [condition])[0]
        else:
            return grad([self.func(condition).mean()], [condition])[0]

    def hessian(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the Hessian matrix at point x.
        :param condition: torch.Tensor
            2D or 1D array on conditions for generator
        :param num_repetitions:
        :return: torch.Tensor
            2D torch tensor with second derivatives
        """
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        if isinstance(num_repetitions, int):
            assert len(condition.size()) == 1
            return hessian_calc(self.func(condition, num_repetitions=num_repetitions), condition)
        else:
            return hessian_calc(self.func(condition).mean(), condition)
