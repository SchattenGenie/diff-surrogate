import torch
import pyro
from pyro import distributions as dist
from pyro import poutine
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
#sns.set()


class YModel(object):
    def __init__(self, x_range=(-10,10), init_mu = 0):
        self.mu_dist = dist.Delta(torch.tensor(float(init_mu), requires_grad=False))
        self.x_dist = dist.Uniform(*x_range)
    
    @staticmethod
    def f(x, a=1, b=1, c=2):
        return a + b * x
    @staticmethod
    def g(x, d=1):
        return d * x
    
    def sample(self, sample_size=1):
        mu = pyro.sample('mu', self.mu_dist, torch.Size([sample_size]))
        if mu.size() == torch.Size([]):
            size = [1]
        else:
            size = mu.size()
        X = pyro.sample('X', self.x_dist, size)

        latent_x = pyro.sample('latent_x', dist.Normal(X, 1))#.double()
        latent_x = self.f(latent_x)

        latent_mu = self.g(mu)#.double()
        return pyro.sample('y', dist.Normal(latent_x + latent_mu, 1))
    
    def make_condition_sample(self, data):
        self.condition_sample = poutine.condition(self.sample, data=data)
    
    def condition_sample(self, size=1):
        return self.condition_sample(size)
    
    
def R(ys: torch.tensor, Y_0=-5):
    return (ys - Y_0).pow(2).mean()    