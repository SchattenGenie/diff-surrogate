import numpy as np
from tqdm import trange
import torch

def sample_noise(N, NOISE_DIM):
    return np.random.normal(size=(N,NOISE_DIM)).astype(np.float32)

def iterate_minibatches(X, batchsize, y=None):
    perm = np.random.permutation(X.shape[0])
    
    for start in trange(0, X.shape[0], batchsize):
        end = min(start + batchsize, X.shape[0])
        if y is None:
            yield X[perm[start:end]]
        else:
            yield X[perm[start:end]], y[perm[start:end]]
            
def generate_data(y_sampler, device, n_samples, mu_range=(-5, 5)):
    mus = ((mu_range[0] - mu_range[1]) * (torch.rand(n_samples)) + mu_range[1]).to(device)
    xs = y_sampler.x_dist.sample([n_samples]).to(device)

    y_sampler.make_condition_sample({'mu': mus, 'X':xs})
    
    data = y_sampler.condition_sample().detach().to(device)
    return data.reshape(-1,1), torch.stack([mus, xs], dim=1)            