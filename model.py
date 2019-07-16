import torch
import pyro
from pyro import distributions as dist
from pyro import poutine
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
#sns.set()


class YModel(object):
    def __init__(self, x_range=(-10,10), init_mu=torch.tensor(0.)):
        self.mu_dist = dist.Delta(init_mu)
        self.x_dist = dist.Uniform(*x_range)
        #self.x_dist = dist.Delta(torch.tensor(float(0)))
    @staticmethod
    def f(x, a=0, b=1, c=2):
        return a + b * x
    @staticmethod
    def g(x, d=2):
        #return -7 + x ** 2 / 10 + x ** 3 / 100
        #return d * torch.sin(x)
        return torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        #return x
    
    def std_val(self, x):
        return 0.1 + torch.abs(x) * 0.5
    
    def sample(self, sample_size=1):
        mu = pyro.sample('mu', self.mu_dist, torch.Size([sample_size]))
        if mu.size() == torch.Size([]):
            size = [1]
        else:
            size = mu.size()
        X = pyro.sample('X', self.x_dist, torch.Size(size))

        latent_x = pyro.sample('latent_x', dist.Normal(X, 1))
        latent_x = self.f(latent_x)

        latent_mu = self.g(mu)
        return pyro.sample('y', dist.Normal(latent_x + latent_mu, self.std_val(latent_x)))
        #return pyro.sample('y', dist.Normal(latent_x, 1)).float()
        #return pyro.sample('y', dist.Normal(latent_x, self.std_val(latent_x))).float()
    
    def make_condition_sample(self, data):
        self.condition_sample = poutine.condition(self.sample, data=data)
    
    def condition_sample(self, size=1):
        return self.condition_sample(size)

class OptLoss(object):
    def __init__(self):
        pass
    
    @staticmethod
    def R(ys: torch.tensor, Y_0=-5):
        return (ys - Y_0).pow(2).mean(dim=1)
    @staticmethod
    def SigmoidLoss(ys: torch.tensor, left_bound, right_bound):
        return -torch.mean(torch.sigmoid(ys - left_bound) - torch.sigmoid(ys - right_bound), dim=1)


