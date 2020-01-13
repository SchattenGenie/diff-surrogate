from comet_ml import Experiment
import traceback
import sys
import os
import click
import torch
import numpy as np
sys.path.append('../')
sys.path.append('./RegressionNN')
from typing import List, Union
from model import YModel, RosenbrockModel, MultimodalSingularityModel, GaussianMixtureHumpModel, \
                  LearningToSimGaussianModel, SHiPModel, BernoulliModel, FullSHiPModel,\
                  ModelDegenerate, ModelInstrict, Hartmann6, \
                  RosenbrockModelInstrict, RosenbrockModelDegenerate, RosenbrockModelDegenerateInstrict
from ffjord_ensemble_model import FFJORDModel as FFJORDEnsembleModel
from ffjord_model import FFJORDModel
from gmm_model import GMMModel
from gan_model import GANModel
from linear_model import LinearModelOnPsi
from optimizer import *
from logger import SimpleLogger, CometLogger, GANLogger, RegressionLogger
from base_model import BaseConditionalGenerationOracle, ShiftedOracle
from constraints_utils import make_box_barriers, add_barriers_to_oracle
from experience_replay import ExperienceReplay, ExperienceReplayAdaptive
from adaptive_borders import AdaptiveBorders
from trust_region import TrustRegion
from base_model import average_block_wise
from RegressionNN.regression_model import RegressionModel, RegressionRiskModel


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


def end_to_end_training(
        epochs: int,
        model_cls: BaseConditionalGenerationOracle,
        optimizer_cls: BaseOptimizer,
        optimized_function_cls: BaseConditionalGenerationOracle,
        logger: BaseLogger,
        model_config: dict,
        optimizer_config: dict,
        n_samples_per_dim: int,
        step_data_gen: float,
        n_samples: int,
        current_psi: Union[List[float], torch.tensor],
        experiment=None
):
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
    :param reuse_model:
    :param reuse_optimizer:
    :param finetune_model:
    :param shift_model:

    :return:
    """
    gan_logger = GANLogger(experiment)
    # gan_logger = RegressionLogger(experiment)
    # gan_logger = None

    y_sampler = optimized_function_cls(device=device, psi_init=current_psi)
    model = model_cls(y_model=y_sampler, **model_config, logger=gan_logger).to(device)

    trust_region = TrustRegion()

    optimizer = optimizer_cls(
        oracle=model,
        x=current_psi,
        **optimizer_config
    )
    print(model_config)
    exp_replay = ExperienceReplay(
        psi_dim=model_config['psi_dim'],
        y_dim=model_config['y_dim'],
        x_dim=model_config['x_dim'],
        device=device
    )
    weights = None
    logger.log_performance(
        y_sampler=y_sampler,
        current_psi=current_psi,
        n_samples=n_samples
    )
    for epoch in range(epochs):
        # generate new data sample
        # condition
        x, condition, conditions_grid, r_grid = y_sampler.generate_local_data_lhs_normal(
            n_samples_per_dim=n_samples_per_dim,
            sigma=adaptive_border.sigma,
            current_psi=current_psi,
            n_samples=n_samples)
        exp_replay.add(y=x, condition=condition)
        x, condition = exp_replay.extract(psi=current_psi, sigma=adaptive_border.sigma)
        print(x.shape, condition.shape)
        print(
            condition[:, :model_config['psi_dim']].std(dim=0).detach().cpu().numpy(),
            np.percentile(condition[:, :model_config['psi_dim']].detach().cpu().numpy(), q=[5, 95], axis=0)
        )
        model = model_cls(y_model=y_sampler, **model_config, logger=gan_logger).to(device)
        model.fit(x, condition=condition, weights=weights)
        model.eval()
        optimizer.update(oracle=model, x=current_psi, step=step_data_gen)
        current_psi, step_data_gen, optimizer, history = trust_region.step(
            y_model=y_sampler,
            model=model,
            previous_psi=current_psi,
            step=step_data_gen,
            optimizer_config=optimizer_config,
            optimizer=optimizer
        )
        print("New step data gen:", step_data_gen)
        if type(y_sampler).__name__ in ['SimpleSHiPModel', 'SHiPModel', 'FullSHiPModel']:
            current_psi = torch.clamp(current_psi, 1e-5, 1e5)
        try:
            # logging optimization, i.e. statistics of psi
            # logger.log_grads(model, y_sampler, current_psi, n_samples_per_dim, log_grad_diff=False)
            logger.log_performance(y_sampler=y_sampler, current_psi=current_psi, n_samples=n_samples)
            logger.log_optimizer(optimizer)
            # too long for ship...
            """
            if not isinstance(y_sampler, SHiPModel):
                logger.log_oracle(oracle=model,
                                  y_sampler=y_sampler,
                                  current_psi=current_psi,
                                  step_data_gen=step_data_gen,
                                  num_samples=200)
            """
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            # raise
        torch.cuda.empty_cache()
    logger.func_saver.join()
    return None


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--optimizer', type=str, default='GradientDescentOptimizer')
@click.option('--logger', type=str, default='CometLogger')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--epochs', type=int, default=10000)
@click.option('--n_samples', type=int, default=10)
@click.option('--lr', type=float, default=1e-1)
@click.option('--step_data_gen', type=float, default=0.1)
@click.option('--n_samples_per_dim', type=int, default=3000)
@click.option('--init_psi', type=str, default="0., 0.")
def main(model,
         optimizer,
         logger,
         optimized_function,
         project_name,
         work_space,
         tags,
         model_config_file,
         optimizer_config_file,
         epochs,
         n_samples,
         step_data_gen,
         n_samples_per_dim,
         lr,
         init_psi,
         ):
    model_config = getattr(__import__(model_config_file), 'model_config')
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)
    model_config['psi_dim'] = psi_dim
    optimizer_config['x_step'] = step_data_gen
    optimizer_config['lr'] = lr

    optimized_function_cls = str_to_class(optimized_function)
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
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('line_search_options', {}).items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('optim_params', {}).items()}
    )
    # experiment.log_asset("./gan_model.py", overwrite=True)
    # experiment.log_asset("./optim.py", overwrite=True)
    # experiment.log_asset("./train.py", overwrite=True)
    # experiment.log_asset("../model.py", overwrite=True)

    logger = str_to_class(logger)(experiment)
    print("Using device = {}".format(device))

    end_to_end_training(
        epochs=epochs,
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        optimized_function_cls=optimized_function_cls,
        logger=logger,
        model_config=model_config,
        optimizer_config=optimizer_config,
        current_psi=init_psi,
        n_samples_per_dim=n_samples_per_dim,
        step_data_gen=step_data_gen,
        n_samples=n_samples,
        experiment=experiment
    )


if __name__ == "__main__":
    main()

