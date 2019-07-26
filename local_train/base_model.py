from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from hessian import hessian as hessian_calc
import sys
sys.path.append("../")
from model import OptLoss


class BaseConditionalGeneratorModel(nn.Module, ABC):
    """
    Base class for implementation of conditional generation model.
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
    def fit(self, x, condition):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('fit is not implemented.')

    @abstractmethod
    def loss(self, x, condition):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('loss is not implemented.')

    @abstractmethod
    def generate(self, condition):
        """
        Generates samples for given conditions
        """
        raise NotImplementedError('predict is not implemented.')

    @abstractmethod
    def log_density(self, x, condition):
        """
        Computes log density for given conditions and x
        """
        raise NotImplementedError('log_density is not implemented.')


class BaseConditionalGenerationOracle(BaseConditionalGeneratorModel, ABC):
    """
    Base class for implementation of loss oracle.
    """
    def func(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the value of function at point x.
        """

        if isinstance(num_repetitions, int):
            assert len(condition.size()) == 1
            conditions = condition.repeat(num_repetitions, 1)
            x = self.generate(conditions)
        else:
            x = self.generate(condition)
        loss = OptLoss.SigmoidLoss(x, 5, 10).mean()
        return loss

    def grad(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the gradient at point x.
        If num_repetitions is not None then condition assumed
        to be 1-d tensor, which would be repeated num_repetitions times
        :param condition: torch.Tensor
            2D or 1D array on conditions for generator
        :param num_repetitions:
        :return: torch.Tensor
            1D torch tensor
        """
        condition = condition.detach().clone()
        condition.requires_grad_(True)
        if isinstance(num_repetitions, int):
            assert len(condition.size()) == 1
            conditions = condition.repeat(num_repetitions, 1)
            return grad([self.func(conditions)], [condition])[0]
        else:
            return grad([self.func(condition)], [condition])[0]

    def hessian(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the Hessian matrix at point x.
        :param condition: torch.Tensor
            2D or 1D array on conditions for generator
        :param num_repetitions:
        :return: torch.Tensor
            2D torch tensor with second derivatives
        """
        condition = condition.detach().clone()
        condition.requires_grad_(True)
        if isinstance(num_repetitions, int):
            assert len(condition.size()) == 1
            conditions = condition.repeat(num_repetitions, 1)
            return hessian_calc(self.func(conditions), condition)
        else:
            return hessian_calc(self.func(condition), condition)
