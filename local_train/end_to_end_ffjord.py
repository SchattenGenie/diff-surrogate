from comet_ml import Experiment
import os
import sys
import math
sys.path.append("../")

import torch
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from model import YModel
from gan_model import Generator, Discriminator, WSDiscriminator, GANLosses,Net
from utils import sample_noise, iterate_minibatches, generate_data
from utils import DistPlotter
from train import GANTrainingUtils
from optim import find_psi, InputOptimisation, make_figures
sys.path.append('./ffjord_cond_bias/')
import ffjord.lib.utils as utils
from ffjord.lib.visualize_flow import visualize_transform
import ffjord.lib.layers.odefunc as odefunc
from ffjord.train_misc import standard_normal_logprob
from ffjord.train_misc import count_nfe, count_parameters, count_total_time
from ffjord.train_misc import add_spectral_norm, spectral_norm_power_iteration
from ffjord.train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from ffjord.train_misc import build_model_tabular
import lib.layers as layers

from pyro import distributions as dist
import numpy as np

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']

print('SOLVERS: ', SOLVERS)

print(odefunc.NONLINEARITIES)

def set_cnf_options(model, solver, rademacher, residual, atol=1e-3, rtol=1e-3):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = solver
            module.atol = atol
            module.rtol = rtol

            # If using fixed-grid adams, restrict order to not be too high.
            if solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

        if isinstance(module, layers.ODEfunc):
            module.rademacher = rademacher
            module.residual = residual

    model.apply(_set)
    
# layer_type - ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
def build_model_tabular(dims=2,
                        condition_dim=2,
                        layer_type='concatsquash', 
                        nonlinearity='relu', 
                        residual=False, 
                        rademacher=False,
                        train_T=True,
                        solver='dopri5',
                        time_length=0.1,
                        divergence_fn='brute_force', # ["brute_force", "approximate"]
                        hidden_dims=(32, 32), 
                        num_blocks=1, batch_norm=False, 
                        bn_lag=0, regularization_fns=None):


    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            condition_dim=condition_dim,
            strides=None,
            conv=False,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=divergence_fn,
            residual=residual,
            rademacher=rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=time_length,
            train_T=train_T,
            regularization_fns=regularization_fns,
            solver=solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    if batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=bn_lag) for _ in range(num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(model, solver, rademacher, residual)

    return model

import math

def standard_normal_logprob(z, data):
    logZ = -0.5 * math.log(2 * math.pi)
    data_ref = torch.zeros_like(z)
    zpow = (z - data_ref).pow(2)
    return logZ - zpow / 2


def compute_loss(model, data, condition, batch_size=None):
    zero = torch.zeros(data.shape[0], 1).to(data.device)
    z, delta_logp = model(data, zero, condition=condition)

    # compute log q(z)
    logpz = standard_normal_logprob(z, data).sum(1, keepdim=True)

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


def get_transforms(model):

    def sample_fn(z, condition, logpz=None):
        if logpz is not None:
            return model(z, condition=condition, logpz=logpz, reverse=True)
        else:
            return model(z, condition=condition, reverse=True)

    def density_fn(x, condition, logpx=None):
        if logpx is not None:
            return model(x, condition=condition, logpz=logpx, reverse=False)
        else:
            return model(x, condition=condition, reverse=False)

    return sample_fn, density_fn

import warnings
warnings.filterwarnings("ignore")



TASK = int(sys.argv[1])
NOISE_DIM = int(sys.argv[2])
exp_tags = sys.argv[3].split("*")
n_samples_per_dim = int(sys.argv[4])
n_d_train = int(sys.argv[5])
batch_size = int(sys.argv[6])
learning_rate = float(sys.argv[7])
INSTANCE_NOISE = bool(int(sys.argv[8]))
CUDA_DEVICE = sys.argv[9]
# snapshot_name = sys.argv[9]

os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)
os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64'

INST_NOISE_STD = 0.3

PROJ_NAME = "end_to_end"
# PATH = "./snapshots/" + PROJ_NAME + "_" + snapshot_name + ".tar"
PATH = os.path.expanduser("~/data/diff_gen_data/n_samples_per_dim/{}".format(n_samples_per_dim))
if not os.path.exists(os.path.expanduser(PATH)):
    os.mkdir(os.path.expanduser(PATH))

hyper_params = {
    "TASK": TASK,
    "batch_size": batch_size,
    "NOISE_DIM": NOISE_DIM,
    "num_epochs": 20,
    "learning_rate": learning_rate,
    "n_samples_per_dim": n_samples_per_dim,
    "n_d_train": n_d_train,
    "INST_NOISE_STD": INST_NOISE_STD,
    "INSTANCE_NOISE": INSTANCE_NOISE,
    # "mu_range": (-10, 11),
    "mu_dim": 2,
    "x_dim": 1,
    "optim_epoch": 1000,
    "grad_step": 2,
#     "random_step_std": 0.1,
    "n_lhc_samples": 5
}

# if len(sys.argv) == 11:
#     from comet_ml import API
#     import comet_ml
#     import io
#
#     comet_api = API()
#     comet_api.get()
#     exp = comet_api.get("shir994/2d-convergence/{}".format(sys.argv[10]))
#     keys = hyper_params.keys()
#     hyper_params = {}
#     for param in exp.parameters:
#         if param["name"] in keys:
#             if param["name"] == "INSTANCE_NOISE":
#                 hyper_params[param["name"]] = param["valueMin"] == 'true'
#             else:
#                 hyper_params[param["name"]] = eval(param["valueMin"])
#     asset_id = \
#     [exp_a['assetId'] for exp_a in exp.asset_list if exp_a['fileName'] == f"{hyper_params['data_size']}_980.tar"][0]
#     params = exp.get_asset(asset_id)
#     state_dict = torch.load(io.BytesIO(params))
#     hyper_params['num_epochs'] = 5000

experiment = Experiment(project_name=PROJ_NAME, workspace="shir994")
experiment.log_parameters(hyper_params)
experiment.add_tags(exp_tags)

device = torch.device("cuda", 0)
TASK = hyper_params['TASK']
experiment.log_asset("./gan_nets.py", overwrite=True)
experiment.log_asset("./optim.py", overwrite=True)
experiment.log_asset("./train.py", overwrite=True)
experiment.log_asset("../model.py", overwrite=True)

# generator = Generator(hyper_params['NOISE_DIM'], out_dim=1,
#                       X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)
# if TASK == 4:
#     discriminator = WSDiscriminator(in_dim=1,
#                                     X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)
# else:
#     discriminator = Discriminator(in_dim=1,
#                                   X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)

# if len(sys.argv) == 11:
#     generator.load_state_dict(state_dict['gen_state_dict'])
#     discriminator.load_state_dict(state_dict['dis_state_dict'])
#     g_optimizer.load_state_dict(state_dict['genopt_state_dict'])
#     d_optimizer.load_state_dict(state_dict['disopt_state_dict'])

y_sampler = YModel()
gan_training = GANTrainingUtils(GANLosses, TASK, device, hyper_params, experiment, y_sampler, PATH, INSTANCE_NOISE)


def end_to_end_training(current_psi):
    r_values = []
    psi_values = []
    try:
        with experiment.train():
            total_epoch_counter = [0]
            for optim_epoch in range(hyper_params["optim_epoch"]):         
                generator = Generator(hyper_params['NOISE_DIM'], out_dim=1,
                                      X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)
                if TASK == 4:
                    discriminator = WSDiscriminator(in_dim=1,
                                                    X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)
                else:
                    discriminator = Discriminator(in_dim=1,
                                                  X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)                
                gan_training.train_gan(generator, discriminator, current_psi, total_epoch_counter)

                io_model = InputOptimisation(generator)             
                psi_vals, losses = find_psi(device, NOISE_DIM, io_model, y_sampler, current_psi,
                                            lr=5000., average_size=1000, n_iter=1, use_true=False)

                current_psi = torch.Tensor(psi_vals)
                psi_values.append(psi_vals)
                r_values.append(losses[0])

                f = make_figures(r_values, np.array(psi_values))
                experiment.log_figure("psi_dynamic", f, overwrite=True)
                plt.close(f)
                
                n_psi = 2000
                average_size = 1000
                fixed_noise = torch.Tensor(sample_noise(n_psi * average_size,
                                                        hyper_params['NOISE_DIM'])).to(device)
                               
                dist_plotter = DistPlotter(y_sampler, generator, fixed_noise, device, mu_dim=hyper_params['mu_dim'])
                f, g = dist_plotter.draw_grads_and_losses(current_psi.view(-1),
                                                          psi_size=n_psi, average_size=average_size,
                                                          step=hyper_params['grad_step'])
                torch.cuda.empty_cache()
                experiment.log_figure("grads_{}".format(optim_epoch), f, overwrite=False)
                experiment.log_figure("loss_{}".format(optim_epoch), g, overwrite=False)
                plt.close(f)
                plt.close(g)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    current_psi = torch.Tensor([1., -1.]).reshape(1,-1)
    end_to_end_training(current_psi)
