from comet_ml import Experiment
import os
import sys
sys.path.append("..")

TASK = int(sys.argv[1])
NOISE_DIM = int(sys.argv[2])
exp_tag = sys.argv[3]
data_size = int(sys.argv[4])

PROJ_NAME = "gan_data_size"
PATH = "./snapshots/" + PROJ_NAME + "_" + str(data_size) + ".tar"

hyper_params = {
    "TASK": TASK,
    "batch_size": 128, # initially was 64
    "NOISE_DIM": NOISE_DIM,
    "num_epochs": 1000,
    "learning_rate": 0.001,
    "data_size": data_size
}
experiment = Experiment(project_name=PROJ_NAME, workspace="shir994")
experiment.log_parameters(hyper_params)
experiment.add_tag(exp_tag)

os.environ['CUDA_VISIBLE_DEVICES']= '3'
os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64'

from model import YModel, R
from gan import Generator, Discriminator, WSDiscriminator, GANLosses
from metrics import Metrics
from utils import sample_noise, iterate_minibatches, generate_data

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import numpy as np
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda", 0)
TASK = hyper_params['TASK']


generator = Generator(hyper_params['NOISE_DIM'], out_dim = 1).to(device)
if TASK == 4:
    discriminator = WSDiscriminator(in_dim=1).to(device)
else:
    discriminator = Discriminator(in_dim=1).to(device)

g_optimizer = optim.Adam(generator.parameters(),     lr=hyper_params['learning_rate'], betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=hyper_params['learning_rate'], betas=(0.5, 0.999))

y_sampler = YModel()
fixed_noise = torch.Tensor(sample_noise(10000, hyper_params['NOISE_DIM'])).to(device)
data, inputs = generate_data(y_sampler, device, data_size, mu_range=(-30, 30))
train_fixed_noise = torch.Tensor(sample_noise(data.shape[0], hyper_params['NOISE_DIM'])).to(device)
metric_calc = Metrics((-50, 50), 100)

def draw_conditional_samples(mu_range, x_range):
    f = plt.figure(figsize=(21,16))

    for i in range(4):
        x = torch.Tensor([float(x_range[i])] * fixed_noise.shape[0]).to(device)
        for j in range(4):
            mu = torch.Tensor([float(mu_range[j])] * fixed_noise.shape[0]).to(device)
            plt.subplot(4,4, i*4 + j + 1)

            y_sampler.make_condition_sample({'mu': mu, 'X':x})
            data = y_sampler.condition_sample().detach().cpu().numpy()

            plt.hist(data, bins=100, density=True, label='true');
            plt.hist(generator(fixed_noise, torch.stack([mu,x],dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');
            plt.grid()
            plt.legend()
            if j == 0:
                plt.ylabel("x={:.3f}".format(x[0].item()), fontsize=19)
            if i == 0:
                plt.title("mu={:.3f}".format(mu[0].item()), fontsize=19)            
    return f

def draw_mu_samples(mu_range):
    f = plt.figure(figsize=(21,16))
    for i in range(4):
        for j in range(3):
            plt.subplot(4,3, i*3 + j + 1)
            mu = torch.tensor([float(mu_range[i*3 + j])] * fixed_noise.shape[0])
            y_sampler.make_condition_sample({'mu': mu})

            x = y_sampler.x_dist.sample(mu.shape).to(device)
            mu = mu.to(device)

            plt.hist(y_sampler.condition_sample().cpu().numpy(), bins=100, density=True, label='true');
            plt.hist(generator(fixed_noise, torch.stack([mu,x],dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');
            plt.grid()
            plt.legend()
            plt.title("mu={:.3f} ".format(mu_range[i*3 + j])) 
    return f

def calculate_validation_metrics(mu_range, epoch):
    js = []
    ks = []
    for _mu in mu_range:
        mu = torch.tensor([float(_mu)] * fixed_noise.shape[0])
        y_sampler.make_condition_sample({'mu': mu})

        x = y_sampler.x_dist.sample(mu.shape).to(device)
        mu = mu.to(device)

        js.append(metric_calc.compute_JS(y_sampler.condition_sample().cpu(),
                  generator(fixed_noise, torch.stack([mu,x],dim=1)).detach().cpu()).item())
        ks.append(metric_calc.compute_KSStat(y_sampler.condition_sample().cpu().numpy(),
                  generator(fixed_noise, torch.stack([mu,x],dim=1)).detach().cpu().numpy()).item())        

    train_data_js = metric_calc.compute_JS(data.cpu(), generator(train_fixed_noise, inputs).detach().cpu())
    train_data_ks = metric_calc.compute_KSStat(data.cpu().numpy(),
                                               generator(train_fixed_noise, inputs).detach().cpu().numpy())

    experiment.log_metric("average_mu_JS", np.mean(js), step=epoch)
    experiment.log_metric("train_data_JS", train_data_js, step=epoch)
    
    experiment.log_metric("average_mu_KS", np.mean(ks), step=epoch)
    experiment.log_metric("train_data_KS", train_data_ks, step=epoch)    
    
    
def run_training():

    # ===========================
    # IMPORTANT PARAMETER:
    # Number of D updates per G update
    # ===========================
    k_d, k_g = 4, 1

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
                        inp_data = input_data.to(device)
                        data_gen = generator(noise, inputs_batch)

                        loss = gan_losses.d_loss(discriminator(data_gen, inputs_batch),
                                                discriminator(inp_data, inputs_batch))
                        if TASK == 4:
                            grad_penalty = gan_losses.calc_gradient_penalty(discriminator,
                                                                            data_gen.data,
                                                                            inputs_batch.data,
                                                                            inp_data.data)
                            loss += grad_penalty

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
                        loss = gan_losses.g_loss(discriminator(data_gen, inputs_batch))
                        g_optimizer.zero_grad()
                        loss.backward()
                        g_optimizer.step()
                    gen_epoch_loss.append(loss.item())
                
                experiment.log_metric("d_loss", np.mean(dis_epoch_loss), step=epoch)
                experiment.log_metric("g_loss", np.mean(gen_epoch_loss), step=epoch)
                
                calculate_validation_metrics(list(range(-30, 30, 2)), epoch)
                
                if epoch % 20 == 0:
                    mu_range = list(range(-10, 10, 4))
                    x_range = list(range(1, 13, 3))
                    f = draw_conditional_samples(mu_range, x_range)
                    experiment.log_figure("conditional_samples", f)
                    plt.close(f)

                    mu_range = list(range(-10, 13, 2))
                    f = draw_mu_samples(mu_range)
                    experiment.log_figure("mu_samples", f)
                    plt.close(f)
                    
                    torch.save({
                        'gen_state_dict': generator.state_dict(),
                        'dis_state_dict': discriminator.state_dict(),
                        'genopt_state_dict': g_optimizer.state_dict(),
                        'disopt_state_dict': d_optimizer.state_dict(),
                        'epoch': epoch
                        }, PATH)
                    experiment.log_asset(PATH, overwrite=True)
                
    except KeyboardInterrupt:
        pass
    
if __name__ == "__main__":
    run_training()