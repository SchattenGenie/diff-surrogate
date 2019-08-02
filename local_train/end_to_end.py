from comet_ml import Experiment
import sys
import os
import click
import torch
import numpy as np
sys.path.append('../')
from typing import List, Union
from utils import generate_local_data_lhs
from model import YModel
from ffjord_model import FFJORDModel
from gan_model import GANModel
from optimizer import *
from logger import SimpleLogger, CometLogger
from base_model import BaseConditionalGenerationOracle


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


def end_to_end_training(epochs: int,
                        model_cls: BaseConditionalGenerationOracle,
                        optimizer_cls: BaseOptimizer,
                        logger: BaseLogger,
                        model_config: dict,
                        optimizer_config: dict,
                        n_samples_per_dim: int,
                        step_data_gen: float,
                        n_samples: int,
                        current_psi: Union[List[float], torch.tensor]):
    """

    :param epochs: int
        number of local training steps to perfomr
    :param model_cls: BaseConditionalGenerationOracle
        model that is able to generate samples and calculate loss function
    :param optimizer_cls: BaseOptimizer
    :param logger: BaseLogger
    :param model_config: dict
    :param optimizer_config: dict
    :param n_samples_per_dim: int
    :param step_data_gen: float
    :param n_samples: int
    :param current_psi:
    :return:
    """
    y_sampler = YModel(device=device, psi_init=current_psi)
    for epoch in range(epochs):
        # generate new data sample
        x, condition = y_sampler.generate_local_data_lhs(
            n_samples_per_dim=n_samples_per_dim,
            step=step_data_gen,
            current_psi=current_psi,
            n_samples=n_samples)
        print(x.shape, condition.shape)
        # at each epoch re-initialize and re-fit
        model = model_cls(y_model=y_sampler, **model_config).to(device)
        model.fit(x, condition=condition)

        # find new psi
        optimizer = optimizer_cls(oracle=model,
                                  x=current_psi,
                                  **optimizer_config)
        current_psi, status, history = optimizer.optimize()
        print(current_psi, status)
        # logging optimization, i.e. statistics of psi
        logger.log_performance(y_sampler=y_sampler, current_psi=current_psi)
        logger.log_optimizer(optimizer)
        logger.log_oracle(oracle=model,
                          y_sampler=y_sampler,
                          current_psi=current_psi,
                          step_data_gen=step_data_gen)

    return xs


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--optimizer', type=str, default='GradientDescentOptimizer')
@click.option('--logger', type=str, default='CometLogger')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--epochs', type=int, default=500)
@click.option('--n_samples', type=int, default=10)
@click.option('--step_data_gen', type=float, default=1.)
@click.option('--n_samples_per_dim', type=int, default=3000)
@click.option('--init_psi', type=str, default="0., 0.")
def main(model,
         optimizer,
         logger,
         project_name,
         work_space,
         tags,
         model_config_file,
         optimizer_config_file,
         epochs,
         n_samples,
         step_data_gen,
         n_samples_per_dim,
         init_psi="0., 0."
         ):
    model_config = getattr(__import__(model_config_file), 'model_config')
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)
    model_config['psi_dim'] = psi_dim

    model_cls = str_to_class(model)
    optimizer_cls = str_to_class(optimizer)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameter('model_type', model)
    experiment.log_parameter('optimizer_type', optimizer)
    experiment.log_parameters(
        {"model_{}".format(key): value for key, value in model_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.items()}
    )
    # experiment.log_asset("./gan_model.py", overwrite=True)
    # experiment.log_asset("./optim.py", overwrite=True)
    # experiment.log_asset("./train.py", overwrite=True)
    # experiment.log_asset("../model.py", overwrite=True)

    logger = str_to_class(logger)(experiment)

    end_to_end_training(
        epochs=epochs,
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        logger=logger,
        model_config=model_config,
        optimizer_config=optimizer_config,
        current_psi=init_psi,
        n_samples_per_dim=n_samples_per_dim,
        step_data_gen=step_data_gen,
        n_samples=n_samples
    )


if __name__ == "__main__":
    main()

