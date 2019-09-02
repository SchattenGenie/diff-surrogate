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
from model import YModel
from optimizer import BaseOptimizer
from typing import Callable


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


class ControlVariate(nn.Module):
    def __init__(self, psi_dim):
        super(ControlVariate, self).__init__()
        self.lin1 = nn.Linear(psi_dim, 5)
        self.lin2 = nn.Linear(5, 5)
        self.lin3 = nn.Linear(5, 5)
        self.lin4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x


class VoidOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 control_variate: Callable,
                 lr: float = 1e-1,
                 K: int = 20,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._policy = NormalPolicy()
        self._control_variate = control_variate().to(oracle.device)
        self._sigma = torch.zeros_like(self._x, requires_grad=True)
        self._K = K
        self._optimizer = optim.Adam(
            params=[self._x, self._sigma] + list(self._control_variate.parameters()),
            lr=lr)

    def _update_history(self, init_time):
        self._history['time'].append(
            time.time() - init_time
        )
        self._history['func_evals'].append(
            self._oracle._n_calls - self._previous_n_calls
        )
        self._history['func'].append(
            self._oracle.func(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )
        self._history['grad'].append(
            self._d_k
        )
        self._history['x'].append(
            self._x.detach().cpu().numpy()
        )
        self._history['alpha'].append(
            self._alpha_k
        )
        self._previous_n_calls = self._oracle._n_calls

    def _step(self):
        init_time = time.time()

        self._optimizer.zero_grad()
        for i in range(self.K):
            action = self._policy(self._x, self._sigma)
            r = self._oracle.func(action)
            c = self._control_variate(action)
            log_prob = policy.log_prob(mu=self._x, sigma=self._sigma, x=action.detach()).mean()

            x_grad_1, sigma_grad_1 = grad([log_prob], [self._x, self._sigma], retain_graph=True, create_graph=True)
            x_grad_2, sigma_grad_2 = grad([c], [self._x, self._sigma], retain_graph=True, create_graph=True)

            x_grad = mu_grad_1 * (r - c) + mu_grad_2
            sigma_grad = sigma_grad_1 * (r - c) + sigma_grad_2
            parameters_grad = grad([x_grad.pow(2).sum() + sigma_grad.pow(2).sum()], parameters)
            with torch.no_grad():
                self._x.grad.data += x_grad.clone().detach() / self._K
                self._sigma.grad.data += sigma_grad.clone().detach() / self._K
                for parameter, parameter_grad in zip(parameters, parameters_grad):
                    parameter.grad.data += parameter_grad.clone().detach()

        self._d_k = self._x.grad.copy()
        self._optimizer.step()

        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


@click.command()
@click.option('--logger', type=str, default='CometLogger')
@click.option('--optimized_function', type=str, default='YModel')
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
    optimizer = VoidOptimizer(oracle=y_model, x=init_psi, **optimizer_config)
    optimizer.optimize()
    logger.log_optimizer(optimizer)


if __name__ == "__main__":
    main()