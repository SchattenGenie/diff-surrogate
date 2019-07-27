from comet_ml import Experiment
import sys
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
device = torch.device('cpu')

def str_to_class(classname):
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

    :param epochs:
    :param model_cls:
    :param optimizer_cls:
    :param logger:
    :param model_config:
    :param optimizer_config:
    :param n_samples_per_dim:
    :param step_data_gen:
    :param n_samples:
    :param current_psi:
    :return:
    """
    y_sampler = YModel(device=device)
    for epoch in range(epochs):
        # generate new data sample
        x, condition = y_sampler.generate_local_data_lhs(
            n_samples_per_dim=n_samples_per_dim,
            step=step_data_gen,
            current_psi=current_psi,
            x_dim=1,  # one left hardcoded parameter
            n_samples=n_samples)
        # at each epoch re-initialize and re-fit
        model = model_cls(**model_config)
        model.fit(x, condition=condition)

        # find new psi
        optimizer = optimizer_cls(oracle=model,
                                  x=current_psi,
                                  **optimizer_config)
        current_psi, status, history = optimizer.optimize()

        # logging optimization, i.e. statistics of psi
        logger.log_optimizer(optimizer)
        logger.log_oracle(oracle=model, y_sampler=y_sampler, current_psi=current_psi)

    return xs


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--optimizer', type=str, default='GradientDescentOptimizer')
@click.option('--logger', type=str, default='CometLogger')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma seperated')
@click.option('--epochs', type=int, default=100)
@click.option('--n_samples', type=int, default=5)
@click.option('--step_data_gen', type=float, default=0.5)
@click.option('--n_samples_per_dim', type=int, default=1000)
@click.option('--init_psi', type=str, default="0., 0.")
def main(model,
         optimizer,
         logger,
         project_name,
         work_space,
         tags,
         model_config_file,
         optimizer_config_file,
         epochs=100,
         n_samples=5,
         step_data_gen=0.5,
         n_samples_per_dim=1000,
         init_psi="0., 0."
         ):
    model_config = getattr(__import__(model_config_file), model_config_file)
    optimizer_config = getattr(__import__(optimizer_config_file), optimizer_config_file)
    init_psi = [float(x.strip()) for x in init_psi.split(',')]
    model_cls = str_to_class(model)
    optimizer_cls = str_to_class(optimizer)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameter('model_type', model)
    experiment.log_parameter('optimizer_type', optimizer)
    experiment.log_parameters(model_config)  # TODO: add prefix to not mix lr of mode
    experiment.log_parameters(optimizer_config)  # and lr of optimizer
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
        current_psi=torch.tensor(init_psi).float().to(device),
        n_samples_per_dim=n_samples_per_dim,
        step_data_gen=step_data_gen,
        n_samples=n_samples
    )


if __name__ == "__main__":
    main()

