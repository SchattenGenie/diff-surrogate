from comet_ml import Experiment
import os
import sys
import math
sys.path.append("..")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from pyro import distributions as dist

import numpy as np
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from model import YModel
#from gan import Generator, Discriminator, WSDiscriminator, GANLosses, PsiCompressor
from local_train.gan_model import Generator, Discriminator, WSDiscriminator, GANLosses
from metrics import Metrics
from utils import sample_noise, iterate_minibatches, generate_data, generate_local_data_lhs
from utils import DistPlotter

TASK = int(sys.argv[1])
NOISE_DIM = int(sys.argv[2])
exp_tags = sys.argv[3].split("*")
#data_size = int(sys.argv[4])
n_epochs = int(sys.argv[4])
n_d_train = int(sys.argv[5])
batch_size = int(sys.argv[6])
learning_rate = float(sys.argv[7])
INSTANCE_NOISE = bool(int(sys.argv[8]))
CUDA_DEVICE = sys.argv[9]
#snapshot_name = sys.argv[9]
current_psi = [float(sys.argv[10])] * 10 #map(float, sys.argv[10].split(","))

os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)
os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64'

INST_NOISE_STD = 0.3


PROJ_NAME = "10d_global_lhc"
#PATH = "./snapshots/" + PROJ_NAME + "_" + snapshot_name + ".tar"
PATH = os.path.expanduser("~/data/diff_gen_data/data_size/{}".format(n_epochs))
if not os.path.exists(os.path.expanduser(PATH)):
    os.mkdir(os.path.expanduser(PATH))

hyper_params = {
    "TASK": TASK,
    "batch_size": batch_size,
    "NOISE_DIM": NOISE_DIM,
    "num_epochs": n_epochs,
    "learning_rate": learning_rate,
    #"data_size": data_size,
    "n_d_train": n_d_train,
    "INST_NOISE_STD": INST_NOISE_STD,
    "INSTANCE_NOISE": INSTANCE_NOISE,
    "mu_dim": 10,
    "x_dim": 1,
    "current_psi": torch.Tensor([*current_psi]).reshape(1, -1),
    "grad_step": 10,
    "n_lhc_samples": 50,
    "n_samples_per_dim": 3000
}
hyper_params["mu_range"] = (hyper_params["current_psi"].view(-1) - hyper_params["grad_step"],
                            hyper_params["current_psi"].view(-1) + hyper_params["grad_step"])

# if len(sys.argv) == 11:
#     from comet_ml import API
#     import comet_ml
#     import io
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
#     asset_id = [exp_a['assetId'] for exp_a in exp.asset_list if exp_a['fileName'] == f"{hyper_params['data_size']}_980.tar"][0]
#     params = exp.get_asset(asset_id)
#     state_dict = torch.load(io.BytesIO(params))
#     hyper_params['num_epochs'] = 5000


experiment = Experiment(project_name=PROJ_NAME, workspace="shir994")
experiment.log_parameters(hyper_params)
experiment.add_tags(exp_tags)

device = torch.device("cuda", 0)
TASK = hyper_params['TASK']
experiment.log_asset("./gan.py", overwrite=True)
experiment.log_asset("../model.py", overwrite=True)

#psi_compressed_dim = 2
#psi_compressor = PsiCompressor(hyper_params['mu_dim'], psi_compressed_dim)
                
                
generator = Generator(hyper_params['NOISE_DIM'], out_dim=1,
                      X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)
if TASK == 4:
    discriminator = WSDiscriminator(in_dim=1,
                                    X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)
else:
    discriminator = Discriminator(in_dim=1,
                                  X_dim=hyper_params['x_dim'], psi_dim=hyper_params['mu_dim']).to(device)

g_optimizer = optim.Adam(generator.parameters(),     lr=hyper_params['learning_rate'], betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=hyper_params['learning_rate'], betas=(0.5, 0.999))
#psi_optimizer = optim.SGD(psi_compressor.parameters(), lr=0.01, momentum=0.9)

# if len(sys.argv) == 11:
#     generator.load_state_dict(state_dict['gen_state_dict'])
#     discriminator.load_state_dict(state_dict['dis_state_dict'])
#     g_optimizer.load_state_dict(state_dict['genopt_state_dict'])
#     d_optimizer.load_state_dict(state_dict['disopt_state_dict'])
    

y_sampler = YModel()
fixed_noise = torch.Tensor(sample_noise(10000, hyper_params['NOISE_DIM'])).to(device)
#data, inputs = generate_data(y_sampler, device, data_size, mu_range=hyper_params["mu_range"], mu_dim=hyper_params['mu_dim'])
data, inputs = generate_local_data_lhs(y_sampler, device,
                                       hyper_params["n_samples_per_dim"],
                                       hyper_params["grad_step"],
                                       hyper_params["current_psi"], x_dim=1, n_samples=hyper_params["n_lhc_samples"])
np.save("train_inputs.npy", inputs.cpu().numpy())
experiment.log_asset("./train_inputs.npy", overwrite=True)
os.remove("./train_inputs.npy")

train_fixed_noise = torch.Tensor(sample_noise(data.shape[0], hyper_params['NOISE_DIM'])).to(device)
metric_calc = Metrics((-50, 50), 100)
dist_plotter = DistPlotter(y_sampler, generator, fixed_noise, device, mu_dim=hyper_params['mu_dim'])

def calculate_validation_metrics(mu_range, epoch, points_size=100, sample_size=2000):
    js = []
    ks = []

    if (epoch) % 20 == 0:
        mu = dist.Uniform(*mu_range).sample([points_size])
        x = y_sampler.x_dist.sample([points_size, hyper_params['x_dim']])
        inputs_mu_x = torch.cat([mu, x], dim=1).to(device)
        for index in range(points_size):
            noise = torch.Tensor(sample_noise(sample_size, hyper_params['NOISE_DIM'])).to(device) 
            sample_inputs = inputs_mu_x[index, :].reshape(1,-1).repeat([sample_size, 1])
            gen_samples = generator(noise, sample_inputs).detach().cpu()
        
            y_sampler.make_condition_sample({'mu': sample_inputs[:, :hyper_params["mu_dim"]],
                                             'X': sample_inputs[:, hyper_params["mu_dim"]:]})
            true_samples = y_sampler.condition_sample().cpu()
            js.append(metric_calc.compute_JS(true_samples, gen_samples).item())
            ks.append(metric_calc.compute_KSStat(true_samples.numpy(), gen_samples.numpy()).item())
        experiment.log_metric("average_mu_JS", np.mean(js), step=epoch)
        experiment.log_metric("average_mu_KS", np.mean(ks), step=epoch)
            
    train_data_js = metric_calc.compute_JS(data.cpu(), generator(train_fixed_noise, inputs).detach().cpu())
    train_data_ks = metric_calc.compute_KSStat(data.cpu().numpy(),
                                               generator(train_fixed_noise, inputs).detach().cpu().numpy())

    experiment.log_metric("train_data_JS", train_data_js, step=epoch)    
    experiment.log_metric("train_data_KS", train_data_ks, step=epoch)    
    for order in range(1, 4):
        moment_of_true = metric_calc.compute_moment(data.cpu(), order)
        moment_of_generated = metric_calc.compute_moment(generator(train_fixed_noise, inputs).detach().cpu(),
                                                 order)
        metric_diff = moment_of_true - moment_of_generated
        
        experiment.log_metric("train_data_diff_order_" + str(order), metric_diff)
        experiment.log_metric("train_data_gen_order_" + str(order), moment_of_generated)       
        
def run_training():

    # ===========================
    # IMPORTANT PARAMETER:
    # Number of D updates per G update
    # ===========================
    k_d, k_g = hyper_params["n_d_train"], 1

    gan_losses = GANLosses(TASK, device)

    try:
        with experiment.train():
            for epoch in range(hyper_params['num_epochs']):
                dis_epoch_loss = []
                gen_epoch_loss = []

                for input_data, inputs_batch in iterate_minibatches(data, hyper_params['batch_size'], y=inputs):
                    # Optimize D
                    for _ in range(k_d):
                        # Sample noise
                        noise = torch.Tensor(sample_noise(len(input_data), hyper_params['NOISE_DIM'])).to(device)


                        # Do an update
                        data_gen = generator(noise, inputs_batch)

                        if INSTANCE_NOISE:
                            inp_data += torch.distributions.Normal(0,hyper_params['INST_NOISE_STD']).\
                                        sample(inp_data.shape).to(device)
                            data_gen += torch.distributions.Normal(0, hyper_params['INST_NOISE_STD']).\
                                        sample(data_gen.shape).to(device)

                        loss = gan_losses.d_loss(discriminator(data_gen, inputs_batch),
                                                 discriminator(input_data, inputs_batch))
                        if TASK == 4:
                            grad_penalty = gan_losses.calc_gradient_penalty(discriminator,
                                                                            data_gen.data,
                                                                            inputs_batch.data,
                                                                            input_data.data)
                            loss += grad_penalty

                        if TASK == 5:
                            grad_penalty = gan_losses.calc_zero_centered_GP(discriminator,
                                                                            data_gen.data,
                                                                            inputs_batch.data,
                                                                            inp_data.data)
                            loss -= grad_penalty                            

                        d_optimizer.zero_grad()
                        loss.backward()
                        d_optimizer.step()

                        if TASK == 3:                    
                            for p in discriminator.parameters():
                                p.data.clamp_(clamp_lower, clamp_upper)
                    dis_epoch_loss.append(loss.item())

                    # Optimize G
                    for _ in range(k_g):
                        # Sample noise
                        noise = torch.Tensor(sample_noise(len(input_data), hyper_params['NOISE_DIM'])).to(device)

                        # Do an update
                        data_gen = generator(noise, inputs_batch)
                        if INSTANCE_NOISE:
                            data_gen += torch.distributions.Normal(0, hyper_params['INST_NOISE_STD']).\
                                        sample(data_gen.shape).to(device)
                        loss = gan_losses.g_loss(discriminator(data_gen, inputs_batch))
                        g_optimizer.zero_grad()
                        #psi_optimizer.zero_grad()
                        loss.backward()
                        g_optimizer.step()
                        #psi_optimizer.step()
                    gen_epoch_loss.append(loss.item())
                
                experiment.log_metric("d_loss", np.mean(dis_epoch_loss), step=epoch)
                experiment.log_metric("g_loss", np.mean(gen_epoch_loss), step=epoch)
                
                calculate_validation_metrics(hyper_params["mu_range"], epoch)
                
                if epoch % 20 == 0:
                    mu_range = hyper_params["mu_range"]
                    f = dist_plotter.draw_conditional_samples(mu_range)
                    experiment.log_figure("conditional_samples_{}".format(epoch), f)
                    plt.close(f)

                    mu_range = hyper_params["mu_range"]
                    f = dist_plotter.draw_mu_samples(mu_range)
                    experiment.log_figure("mu_samples_{}".format(epoch), f)
                    plt.close(f)
                    
                    # x_range = (-10,10)
                    # f = dist_plotter.draw_X_samples(x_range)
                    # experiment.log_figure("x_samples_{}".format(epoch), f)
                    # plt.close(f)
                    
#                     f, g = dist_plotter.plot_means_diff(hyper_params["mu_range"], x_range)
#                     experiment.log_figure("mean_diff_x_{}".format(epoch), f)
#                     experiment.log_figure("mean_diff_{}".format(epoch), g)
#                     plt.close(f)
#                     plt.close(g)

#                     f = dist_plotter.draw_mu_2d_samples(hyper_params["mu_range"])
#                     experiment.log_figure("mu_samples_2d_{}".format(epoch), f)
#                     plt.close(f)
                    
                    snapshot_path = os.path.join(PATH, "{}.tar".format(epoch))
                    torch.save({
                        'gen_state_dict': generator.state_dict(),
                        'dis_state_dict': discriminator.state_dict(),
                        'genopt_state_dict': g_optimizer.state_dict(),
                        'disopt_state_dict': d_optimizer.state_dict(),
                        'epoch': epoch
                        }, snapshot_path)
                    experiment.log_asset(snapshot_path, overwrite=True)
                
    except KeyboardInterrupt:
        pass
    
if __name__ == "__main__":
    run_training()