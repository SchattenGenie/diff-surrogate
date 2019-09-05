from comet_ml import Experiment
import sys
import os
import click
import torch
import numpy as np
from typing import Callable
sys.path.append('../')
from typing import List, Union
from logger import SimpleLogger, CometLogger
from num_diff_schemes import compute_gradient_of_vector_function
from base_model import BaseConditionalGenerationOracle
sys.path.append('../..')
from model import YModel, LearningToSimGaussianModel, GaussianMixtureHumpModel, RosenbrockModel
from num_diff_schemes import compute_gradient_of_vector_function
from num_diff_schemes import n_order_scheme, richardson
from optimizer import GradientDescentOptimizer, ConjugateGradientsOptimizer, LBFGSOptimizer, \
                      GPOptimizer, NewtonOptimizer


def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(get_freer_gpu()))
else:
    device = torch.device('cpu')
print("Using device = {}".format(device))


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


class NumericalDifferencesModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 num_repetitions: int,
                 grad_func: Callable,
                 n: int,
                 h: float
                 ):
        super(NumericalDifferencesModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        self._num_repetitions = num_repetitions
        self._n_calls = 0
        self._n = n
        self._h = h
        self._grad_func = grad_func

    @property
    def device(self):
        return self._y_model._device

    def func(self, condition, **kwargs):
        condition = torch.tensor(condition)
        self._n_calls += 1
        return self._y_model.func(condition, num_repetitions=self._num_repetitions)

    def grad(self, condition, **kwargs):
        if not isinstance(condition, list):
            condition = condition.tolist()
        func_func = lambda t: self.func(t).item()
        grad = self._grad_func(f=func_func, x=condition, n=self._n, h=self._h)
        return torch.tensor(grad).float().to(self.device)

    def fit(self):
        pass

    def generate(self):
        pass

    def log_density(self):
        pass

    def loss(self):
        pass


@click.command()
@click.option('--logger', type=str, default='CometLogger')
@click.option('--optimizer', type=str, default='YModel')
@click.option('--optimizer_config_file', type=str, default='optimizer_config_num_diff')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--diff_scheme', type=str, default='n_order_scheme')  # 'richardson'
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--n', type=int, default=3)
@click.option('--num_repetitions', type=int, default=5000)
@click.option('--h', type=float, default=0.2)
@click.option('--init_psi', type=str, default="0., 0.")
def main(
        logger,
        optimized_function,
        optimizer,
        diff_scheme,
        optimizer_config_file,
        project_name,
        work_space,
        tags,
        num_repetitions,
        n,
        h,
        init_psi,
):
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)

    optimized_function_cls = str_to_class(optimized_function)
    optimizer_cls = str_to_class(optimizer)
    diff_scheme_func = str_to_class(diff_scheme)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameter('optimizer_type', optimizer)
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('line_search_options', {}).items()}
    )

    logger = str_to_class(logger)(experiment)
    y_model = optimized_function_cls(device=device, psi_init=init_psi)
    grad_func = lambda f, x, n, h: compute_gradient_of_vector_function(f=f, x=x, n=n, h=h, scheme=diff_scheme_func)
    ndiff = NumericalDifferencesModel(y_model=y_model,
                                      psi_dim=psi_dim,
                                      y_dim=1,
                                      x_dim=1,
                                      n=n,
                                      h=h,
                                      num_repetitions=num_repetitions,
                                      grad_func=grad_func)
    optimizer = optimizer_cls(oracle=ndiff, x=init_psi, **optimizer_config)
    optimizer.optimize()
    logger.log_optimizer(optimizer)




if __name__ == "__main__":
    main()