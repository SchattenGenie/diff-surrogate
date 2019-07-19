import sys
sys.path.append("../")
from copy import deepcopy
from utils import sample_noise
import torch

import torch.nn as nn
from model import OptLoss
import numpy as np
import matplotlib.pyplot as plt


class InputOptimisation(nn.Module):
    def __init__(self, generator_model):
        super().__init__()
        self.gen = deepcopy(generator_model)
        for param in self.gen.parameters():
            param.requires_grad = False
        self.gen.eval()

    def forward(self, noise, inputs):
        return self.gen(noise, inputs)


def find_psi(device, NOISE_DIM, io_model, y_sampler, init_mu, lr = 50., average_size=1000, n_iter=10000, use_true=False):
    mu_optim = init_mu.clone().detach()
    mu_optim = mu_optim.repeat(average_size, 1).to(device)
    mu_optim.requires_grad = True

    losses = []
    m_vals = []
    for _iter in range(n_iter):
        noise = torch.Tensor(sample_noise(average_size, NOISE_DIM)).to(device)
        x = y_sampler.x_dist.sample([average_size, 1]).to(device)
        # Do an update
        if use_true:
            y_sampler.make_condition_sample({"mu": mu_optim, "X": x})
            data_gen = y_sampler.condition_sample()
        else:
            data_gen = io_model(noise, torch.cat([mu_optim, x], dim=1))
        loss = OptLoss.SigmoidLoss(data_gen, 5, 10).mean()
        losses.append(loss.item())
        io_model.zero_grad()
        loss.backward()
        with torch.no_grad():
            mu_optim -= lr * mu_optim.grad.mean(dim=0, keepdim=True)
            mu_optim.grad.zero_()
        m_vals.append(mu_optim[0].detach().cpu().numpy())
    m_vals = np.array(m_vals)
    return m_vals, losses


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def make_figures(losses, m_vals):
    f = plt.figure(figsize=(18,6))

    plt.subplot(1,2,1)
    plt.plot(losses);
    plt.grid()
    plt.ylabel("Loss", fontsize=19)
    plt.xlabel("iter", fontsize=19)
    plt.plot((movingaverage(losses, 50)), c='r')

    plt.subplot(1,2,2)
    for i in range(m_vals.shape[1]):
        plt.plot(m_vals[:,i], label=i);
    plt.grid()
    plt.ylabel("$\mu$", fontsize=19)
    plt.xlabel("iter", fontsize=19)
    plt.legend()
    return f
