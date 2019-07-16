from comet_ml import Experiment
experiment = Experiment(project_name="physics_2d", workspace="shir994")

import os
import sys
#sys.path.append("..")
sys.path.append("../..")

import pandas as pd
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import numpy as np
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from gan import GANLosses, Generator, Discriminator, WSDiscriminator
from utils import iterate_minibatches

my_cmap = plt.cm.jet
my_cmap.set_under('white')

TASK = int(sys.argv[1])
NOISE_DIM = int(sys.argv[2])
exp_tags = sys.argv[3].split("*")
n_d_train = int(sys.argv[4])
batch_size = int(sys.argv[5])
learning_rate = float(sys.argv[6])
INSTANCE_NOISE = bool(int(sys.argv[7]))
CUDA_DEVICE = sys.argv[8]


os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64"
device = torch.device("cuda", 0)


hyper_params = {
    'TASK': TASK,
    "batch_size": batch_size,
    "NOISE_DIM": NOISE_DIM,
    "num_epochs": 1000,
    "learning_rate": learning_rate,
    "n_d_train": n_d_train,
    "INST_NOISE_STD": 0.3,
    "INSTANCE_NOISE": INSTANCE_NOISE,
    "param_dim": 2,
    "x_dim": 4
}
experiment.log_parameters(hyper_params)
experiment.add_tags(exp_tags)
experiment.log_asset("./gan.py", overwrite=True)

INSTANCE_NOISE = hyper_params['INSTANCE_NOISE']
TASK = hyper_params['TASK']
PATH = "./physics_gan.tar"

df = pd.read_csv(os.path.expanduser("~/data/diff_gen_data/physics_data/xz_magnet_opt.csv"), index_col=0)
df = df.sample(frac=1)
data_columns = ["hit_x", "hit_y", "hit_E"]
inputs_columns = ["pid", "start_theta", "start_phi", "start_P", "magn_len", "magn_x"]
data = torch.Tensor(df[data_columns].to_numpy(dtype=np.float32)).to(device)
inputs = torch.Tensor(df[inputs_columns].to_numpy(dtype=np.float32)).to(device)


generator = Generator(hyper_params['NOISE_DIM'], out_dim=3, hidden_dim=100,
                      input_param=hyper_params['param_dim'] + hyper_params['x_dim']).to(device)
if TASK == 4:
    discriminator = WSDiscriminator(in_dim=3, hidden_dim=100,
                                    input_param=hyper_params['param_dim'] + hyper_params['x_dim']).to(device)
else:
    discriminator = Discriminator(in_dim=3, hidden_dim=100,
                              input_param=hyper_params['param_dim'] + hyper_params['x_dim']).to(device)

g_optimizer = optim.Adam(generator.parameters(),     lr=hyper_params['learning_rate'], betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=hyper_params['learning_rate'], betas=(0.5, 0.999))

def sample_noise(N, NOISE_DIM):
    return np.random.normal(size=(N,NOISE_DIM)).astype(np.float32)


def genearte_plot_data(n_samples, magn_len, magn_x):
        theta = torch.empty(size=[n_samples,1]).uniform_(df.start_theta.min(), df.start_theta.max())
        phi = torch.empty(size=[n_samples,1]).uniform_(df.start_phi.min(), df.start_phi.max())
        p = torch.empty(size=[n_samples,1]).uniform_(df.start_P.min(), df.start_P.max())

        pids = torch.distributions.Bernoulli(probs=0.5).sample([n_samples, 1])
        pids[pids == 1] = 13.
        pids[pids == 0] = -13.

        noise = torch.Tensor(sample_noise(n_samples, hyper_params['NOISE_DIM'])).to(device)
        distr = generator(noise, torch.cat([pids, theta, phi, p, magn_len, magn_x], dim=1).to(device)).detach().cpu().numpy()     
        return distr

def draw_hitmap(n_samples=100):
    f = plt.figure(figsize=(21,16))
    for index in range(16):
        plt.subplot(4,4, index + 1)
        
        
        magn_len = torch.empty(size=[1, 1], dtype=torch.float32).uniform_(1, 15).repeat([n_samples, 1])
        magn_x = torch.empty(size=[1, 1], dtype=torch.float32).uniform_(1, 10).repeat([n_samples, 1])
        distr = genearte_plot_data(n_samples, magn_len, magn_x)
        distr = distr[distr[:, 2] > 1e-5]

        plt.hist2d(distr[:,0], distr[:, 1], bins=50, cmap=my_cmap, cmin=1e-10)
        plt.grid()
        plt.colorbar()
        plt.title("len={:.2f}, x={:.2f}".format(magn_len[0,0].item(), magn_x[0,0].item()), fontsize=15)
    return f

def draw_2d_hitmap(n_samples=2000):
    f = plt.figure(figsize=(15,20))
    gs1 = gridspec.GridSpec(15, 10)
    gs1.update(wspace=0.025, hspace=0.05)

    for i in range(1,15):
        for j in range(1,10):
            ax = plt.subplot(gs1[i,j])
            
            magn_len = torch.Tensor([i]).repeat([n_samples, 1])
            magn_x = torch.Tensor([j]).repeat([n_samples, 1])
            distr = genearte_plot_data(n_samples, magn_len, magn_x)
            distr = distr[distr[:, 2] > 1e-5]
            
            plt.hist2d(distr[:, 0], distr[:, 1],
                       bins=50, cmap=my_cmap, cmin=1e-10,
                       range=((-3000, 3000), (-3000, 3000)))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            if j == 1:
                ax.set_ylabel(i, fontsize=15)
            if i == 14:
                ax.set_xlabel(j, fontsize=15)
            plt.tick_params(
                axis='both',       
                which='both',      
                bottom=False,      
                left=False,
                labelbottom=False)
    return f

def draw_energy(n_samples=100):
    f = plt.figure(figsize=(21,16))
    for index in range(16):
        plt.subplot(4,4, index + 1)
        
        
        magn_len = torch.empty(size=[1, 1], dtype=torch.float32).uniform_(1, 15).repeat([n_samples, 1])
        magn_x = torch.empty(size=[1, 1], dtype=torch.float32).uniform_(1, 10).repeat([n_samples, 1])
        distr = genearte_plot_data(n_samples, magn_len, magn_x)
        distr = distr[distr[:, 2] > 1e-5]

        #plt.hist2d(distr[:,0], distr[:, 1], bins=50, cmap=my_cmap, cmin=1e-10)
        plt.hist(distr[:, 2], bins=50)
        plt.grid()
        plt.title("len={:.2f}, x={:.2f}".format(magn_len[0,0].item(), magn_x[0,0].item()), fontsize=15)
    return f

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
                        inp_data = input_data.to(device)
                        data_gen = generator(noise, inputs_batch)

                        if INSTANCE_NOISE:
                            inp_data += torch.distributions.Normal(0,hyper_params['INST_NOISE_STD']).\
                                        sample(inp_data.shape).to(device)
                            data_gen += torch.distributions.Normal(0, hyper_params['INST_NOISE_STD']).\
                                        sample(data_gen.shape).to(device)

                        loss = gan_losses.d_loss(discriminator(data_gen, inputs_batch),
                                                discriminator(inp_data, inputs_batch))
                        if TASK == 4:
                            grad_penalty = gan_losses.calc_gradient_penalty(discriminator,
                                                                            data_gen.data,
                                                                            inputs_batch.data,
                                                                            inp_data.data)
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
                        loss.backward()
                        g_optimizer.step()
                    gen_epoch_loss.append(loss.item())
                
                experiment.log_metric("d_loss", np.mean(dis_epoch_loss), step=epoch)
                experiment.log_metric("g_loss", np.mean(gen_epoch_loss), step=epoch)
                
                
                if epoch % 20 == 0:
                    f = draw_hitmap(n_samples=2000)
                    experiment.log_figure("hitmap_radnom_{}".format(epoch), f)
                    plt.close(f)
                if epoch % 50 == 0:
                    f = draw_2d_hitmap(n_samples=2000)
                    
                    f_name = "{:32x}".format(random.getrandbits(32)) + ".pdf"
                    plt.savefig(f_name)
                    experiment.log_asset(f_name, file_name="hitmap_2d_{}.pdf".format(epoch),
                                         overwrite=False, copy_to_tmp=False)
                    #experiment.log_figure("hitmap_2d_{}".format(epoch), f)
                    plt.close(f)
                    os.remove(f_name)
                    
                    f = draw_energy(n_samples=10000)
                    experiment.log_figure("energy_radnom_{}".format(epoch), f)
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
