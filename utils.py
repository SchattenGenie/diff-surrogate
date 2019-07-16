import numpy as np
from tqdm import trange
import torch
import matplotlib.pyplot as plt
from pyro import distributions as dist

def sample_noise(N, NOISE_DIM):
    return np.random.uniform(size=(N,NOISE_DIM)).astype(np.float32)

def iterate_minibatches(X, batchsize, y=None):
    perm = np.random.permutation(X.shape[0])
    
    for start in range(0, X.shape[0], batchsize):
        end = min(start + batchsize, X.shape[0])
        if y is None:
            yield X[perm[start:end]]
        else:
            yield X[perm[start:end]], y[perm[start:end]]
            
def generate_data(y_sampler, device, n_samples, mu_range=(-5, 5), mu_dim=1, x_dim=1):
    #mus = torch.empty([n_samples, mu_dim]).uniform_(*mu_range).to(device)
    mus = torch.randint(*mu_range, [n_samples, mu_dim], dtype=torch.float32).to(device)
    xs = y_sampler.x_dist.sample(torch.Size([n_samples, x_dim])).to(device)

    y_sampler.make_condition_sample({'mu': mus, 'X':xs})
    
    data = y_sampler.condition_sample().detach().to(device)
    return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)


class DistPlotter(object):
    def __init__(self, y_sampler, generator, noise, device, mu_dim=1, x_dim=1):
        self.y_sampler = y_sampler
        self.generator = generator
        self.fixed_noise = noise
        self.device = device
        self.mu_dim = mu_dim
        self.x_dim = x_dim

    def draw_conditional_samples(self, mu_range):
        f = plt.figure(figsize=(21,16))
        
        mu = dist.Uniform(*mu_range).sample([16, self.mu_dim])
        x = self.y_sampler.x_dist.sample([16, self.x_dim])
        
        for index in range(16):
            plt.subplot(4, 4, index + 1)
            mu_s = mu[index, :].repeat(len(self.fixed_noise), 1).to(self.device)
            x_s = x[index, :].repeat(len(self.fixed_noise), 1).to(self.device)
            self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})
            data = self.y_sampler.condition_sample().detach().cpu().numpy()
            
            plt.hist(data, bins=100, density=True, label='true');
            plt.hist(self.generator(self.fixed_noise, torch.cat([mu_s, x_s], dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');
            plt.grid()
            plt.legend()
            plt.ylabel("x={}".format(x[index, :].cpu().numpy()), fontsize=15)
            plt.title("mu={}".format(mu[index, :].cpu().numpy()), fontsize=15)            
        return f
        

#     def draw_mu_samples(self, mu_range, noise_size=1000, n_samples=1000):
#         f = plt.figure(figsize=(21,16))
#         mu = dist.Uniform(*mu_range).sample([12, self.mu_dim])
#         for index in range(12):
#             plt.subplot(4, 4, index + 1)
#             y_samples = []
#             for _iter in range(n_samples):
#                 mu_s = mu[index, :].repeat(noise_size, 1).to(self.device)
#                 noise = torch.Tensor(sample_noise(noise_size, self.fixed_noise.shape[1])).to(self.device)
#                 x_s = self.y_sampler.x_dist.sample([len(mu_s), self.x_dim]).to(self.device)
#                 y_samples.append(self.generator(noise, torch.cat([mu_s, x_s], dim=1)).mean().item())
            
#             mu_s = mu[index, :].repeat(n_samples, 1).to(self.device)
#             x_s = self.y_sampler.x_dist.sample([len(mu_s), noise_size]).to(self.device)
#             self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})
                
#             plt.hist(self.y_sampler.condition_sample().mean(dim=1).cpu().numpy(), bins=100, density=True, label='true');
#             plt.hist(y_samples,
#                      bins=100, color='g', density=True, alpha=0.5, label='gan');
#             plt.grid()
#             plt.legend()
#             plt.title("mu={}".format(mu[index, :].cpu().numpy()), fontsize=15)      
#         return f

    def draw_mu_samples(self, mu_range, noise_size=1000, n_samples=1000):
        f = plt.figure(figsize=(21,16))
        mu = dist.Uniform(*mu_range).sample([16, self.mu_dim])
        for index in range(16):
            plt.subplot(4, 4, index + 1)
            noise = torch.Tensor(sample_noise(self.fixed_noise.shape[0], self.fixed_noise.shape[1])).to(self.device)
            mu_s = mu[index, :].repeat(self.fixed_noise.shape[0], self.mu_dim).to(self.device)
            x_s = self.y_sampler.x_dist.sample([len(mu_s), self.x_dim]).to(self.device)
            self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})

            plt.hist(self.y_sampler.condition_sample().cpu().numpy(), bins=100, density=True, label='true');
            plt.hist(self.generator(noise, torch.cat([mu_s, x_s], dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');    
            plt.grid()
            plt.legend()
            plt.title("mu={}".format(mu[index, :].cpu().numpy()), fontsize=15);
        return f
            
    def draw_X_samples(self, x_range):
        f = plt.figure(figsize=(21,16))
        x = dist.Uniform(*x_range).sample([12, self.x_dim])
        for index in range(12):
            plt.subplot(4,3, index + 1)
            x_s = x[index, :].repeat(len(self.fixed_noise), 1).to(self.device)
            mu_s = self.y_sampler.mu_dist.sample(torch.Size([len(x_s), self.mu_dim])).to(self.device)
            self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})

            plt.hist(self.y_sampler.condition_sample().cpu().numpy(), bins=100, density=True, label='true');
            plt.hist(self.generator(self.fixed_noise, torch.cat([mu_s,x_s],dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');
            plt.grid()
            plt.legend()
            plt.title("x={}".format(x[index, :].cpu().numpy()), fontsize=15)
        return f
    
    def draw_mu_2d_samples(self, mu_range, noise_size=1000):
        my_cmap = plt.cm.jet
        my_cmap.set_under('white')
        mu = dist.Uniform(*mu_range).sample([5000, 2]).to(self.device)
        
        y = np.zeros([len(mu), 1])
        
        for i in range(len(mu)):
            noise = torch.Tensor(sample_noise(noise_size, self.fixed_noise.shape[1])).to(self.device)
            mu_r = mu[i, :].reshape(1,-1).repeat(noise_size, 1).to(self.device)
            x_r = self.y_sampler.x_dist.sample(torch.Size([len(mu_r), 1])).to(self.device)
            y[i, 0] = self.generator(noise, torch.cat([mu_r,x_r],dim=1)).mean().item()

        f = plt.figure(figsize=(12,6))
        mu = mu.cpu().numpy()
        plt.scatter(mu[:,0], mu[:, 1], c=y[:,0], cmap=my_cmap)
        plt.colorbar()
        return f
    
    def plot_means_diff(self, mu_range, x_range):
        means_diff = []
        for index, mu in enumerate(torch.arange(*mu_range, 1)):
            t_means = []
            g_means = []
            for x in torch.arange(*x_range, 0.5):
                #plt.subplot(5, 4, index + 1)
                mu_s = mu.float().reshape(-1,1).repeat(self.fixed_noise.shape[0], 1).to(self.device)
                noise = torch.Tensor(sample_noise(self.fixed_noise.shape[0], self.fixed_noise.shape[1])).to(self.device)
                x_s = x.float().reshape(-1,1).repeat(self.fixed_noise.shape[0], 1).to(self.device)
                y_samples = self.generator(noise, torch.cat([mu_s, x_s], dim=1)).cpu().detach().numpy()
                self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})
                t_means.append(np.mean(y_samples))
                g_means.append(self.y_sampler.condition_sample().cpu().numpy().mean())
            if index == 10:
                f = plt.figure(figsize=(12,6))
                plt.scatter(np.arange(*x_range, 0.5), t_means, label='g')
                plt.scatter(np.arange(*x_range, 0.5), g_means, label='t')
                plt.legend()
                plt.grid()
            means_diff.append((np.array(g_means) - np.array(t_means)).mean())
        g = plt.figure(figsize=(12,6))
        plt.scatter(np.arange(*mu_range, 1), means_diff)
        plt.xlabel(f"$\mu$", fontsize=19)
        plt.ylabel("means_diff")
        plt.grid();
        return f, g