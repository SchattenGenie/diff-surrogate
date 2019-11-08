import torch
from scipy.stats import chi2


class ExperienceReplay:
    def __init__(self, psi_dim, x_dim, y_dim, device, sphere_cut=False):
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._device = device
        self._sphere_cut = sphere_cut
        self._y = torch.zeros(0, self._y_dim).float().to('cpu')
        self._condition = torch.zeros(0, self._x_dim + self._psi_dim).float().to('cpu')

    def add(self, y, condition):
        self._y = torch.cat([self._y, y.to('cpu')], dim=0)
        self._condition = torch.cat([self._condition, condition.to('cpu')], dim=0)
        return self

    def extract(self, psi, step):
        psi = psi.float().to('cpu')
        if self._sphere_cut:
            mask = ((self._condition[:, :self._psi_dim] - psi).pow(2).sum(dim=1).sqrt() < step)  # sphere
        else:
            mask = ((self._condition[:, :self._psi_dim] - psi).abs() < step).all(dim=1)
        y = (self._y[mask]).to(self._device)
        condition = (self._condition[mask]).to(self._device)
        return y, condition


class ExperienceReplayAdaptive:
    def __init__(self, psi_dim, x_dim, y_dim, device, sphere_cut=False):
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._device = device
        self._sphere_cut = sphere_cut
        self._y = torch.zeros(0, self._y_dim).float().to('cpu')
        self._condition = torch.zeros(0, self._x_dim + self._psi_dim).float().to('cpu')

    def add(self, y, condition):
        self._y = torch.cat([self._y, y.to('cpu')], dim=0)
        self._condition = torch.cat([self._condition, condition.to('cpu')], dim=0)
        return self

    def extract(self, psi, sigma, q=0.95):
        psi = psi.float().to('cpu')
        mask = (((self._condition[:, :self._psi_dim] - psi).pow(2) / sigma.pow(2)).sum(dim=1) < chi2(df=self._psi_dim).ppf(q))
        y = (self._y[mask]).to(self._device)
        condition = (self._condition[mask]).to(self._device)
        return y, condition
