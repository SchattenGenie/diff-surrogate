import torch
import pyro
import numpy as np
from pyro import distributions as dist
from local_train.base_model import BaseConditionalGenerationOracle
from pyro import poutine
import matplotlib.pyplot as plt
from pyDOE import lhs
import seaborn as sns
import lhsmdu
import tqdm


class YModel(BaseConditionalGenerationOracle):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10), y_dim=1,
                 loss=lambda y: OptLoss.SigmoidLoss(y, 5, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=y_dim) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
        self._y_dim = y_dim

    @property
    def _y_model(self):
        return self

    @property
    def device(self):
        return self._device

    @staticmethod
    def f(x, a=0, b=1):
        return a + b * x

    @staticmethod
    def g(x):
        return x.pow(2).sum(dim=1, keepdim=True).sqrt()

    @staticmethod
    def std_val(x):
        return 0.1 + x.abs() * 0.5

    def sample_psi(self, sample_size):
        return pyro.sample('mu', self._psi_dist, torch.Size([sample_size])).to(self.device)

    def sample_x(self, sample_size):
        return pyro.sample('x', self._x_dist, torch.Size([sample_size])).to(self.device).view(-1, self._x_dim)

    def _generate_dist(self, psi, x):
        latent_x = self.f(pyro.sample('latent_x', dist.Normal(x, 1))).to(self.device)
        latent_psi = self.g(psi)
        return dist.Normal(latent_x + latent_psi, self.std_val(latent_x))

    def _generate(self, psi, x):
        return pyro.sample('y', self._generate_dist(psi, x))

    def generate(self, condition):
        psi, x = condition[:, :self._psi_dim], condition[:, self._psi_dim:]
        return self._generate(psi, x)

    def sample(self, sample_size):
        psi = self.sample_psi(sample_size)
        x = self.sample_x(sample_size)
        return self._generate(psi, x)

    def loss(self, y, condition):
        pass

    def fit(self, y, condition):
        pass

    def log_density(self, y, condition):
        psi, x = condition[:, :self._psi_dim], condition[:, self._psi_dim:]
        return self._generate_dist(psi, x).log_prob(y)

    def condition_sample(self):
        raise NotImplementedError("First call self.make_condition_sample")

    def make_condition_sample(self, data):
        self.condition_sample = poutine.condition(self.sample, data=data)

    def generate_local_data(self, n_samples_per_dim, step, current_psi, x_dim=1, std=0.1):
        xs = self.x_dist.sample(
            torch.Size([n_samples_per_dim * 2 * current_psi.shape[1] + n_samples_per_dim, x_dim])).to(self.device)

        mus = torch.empty((xs.shape[0], current_psi.shape[1])).to(self.device)

        iterator = 0
        for dim in range(current_psi.shape[1]):
            for dir_step in [-step, step]:
                random_mask = torch.torch.randn_like(current_psi)
                random_mask[0, dim] = 0
                new_psi = current_psi + random_mask * std
                new_psi[0, dim] += dir_step

                mus[iterator:
                    iterator + n_samples_per_dim, :] = new_psi.repeat(n_samples_per_dim, 1)
                iterator += n_samples_per_dim

        mus[iterator: iterator + n_samples_per_dim, :] = current_psi.repeat(n_samples_per_dim, 1).clone().detach()

        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample().detach().to(self.device)
        return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * n_samples)

        # mus = torch.tensor(lhsmdu.sample(len(current_psi), n_samples,
        #                                  randomSeed=np.random.randint(1e5)).T).float().to(self.device)
        mus = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)
        mus = step * (mus * 2 - 1) + current_psi
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)


class RosenbrockModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class MultimodalSingularityModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10),
                 loss=lambda y: torch.mean(y, dim=1)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=1) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
    @staticmethod
    def g(x):
        return x.abs().sum(dim=1, keepdim=True) * ((-x.pow(2).sin().sum(dim=1, keepdim=True)).exp())


class GaussianMixtureHumpModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=2,
                 loss = lambda y: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    def _generate_dist(self, psi, x):
        return self.mixture_model(psi, x)

    def _generate(self, psi, x):
        return pyro.sample('y', self._generate_dist(psi, x))

    def mixture_model(self, psi, x, K=2):
        locs = pyro.sample('locs', dist.Normal(x * self.three_hump(psi).view(-1, 1), 1.)).to(self.device)
        #scales = pyro.sample('scale', dist.LogNormal(0., 2), torch.Size([len(x)])).view(-1, 1).to(self.device)
        assignment = pyro.sample('assignment', dist.Categorical(torch.abs(psi)))
        return dist.Normal(locs.gather(1, assignment.unsqueeze(1)), 1)

    # Three hump function http://benchmarkfcns.xyz/2-dimensional
    def three_hump(self, y):
        return 2 * y[:, 0] ** 2 - 1.05 * y[:, 0] ** 4 + y[:, 0] ** 6 / 6 + y[:,0] * y[:,1] + y[:, 1] ** 2


class LearningToSimGaussianModel(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_dim=1, y_dim=3):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Delta(torch.Tensor([0]).to(device))
        self._psi_dim = len(psi_init)
        self._device = device
        self.create_test_data()
        self.train_discriminator()
        print("INIT_DONE")

    def create_test_data(self):
        class_size = 250
        self.n_class_params = 12
        self.true_params = {0: {0: {"mean": [-7.5, 0], "var": [0.5, 1e-15]},
                                1: {"mean": [-3, 3.], "var": [1, 0.5]},
                                2: {"mean": [3, -3.], "var": [1, 0.5]}},
                            1: {0: {"mean": [0, 5], "var": [0.5, 1e-15]},
                                1: {"mean": [3, 3.], "var": [1, 0.5]},
                                2: {"mean": [-3, -3.], "var": [1, 0.5]}}}

        # this is just a messy way to induce readability of what parameter means what in the psi tensor
        # and double check everything works with one tensor
        data_generation = []
        n_components = 3
        for class_index in [1, 0]:
            for i in range(n_components):
                data_generation.extend([*self.true_params[class_index][i]["mean"],
                                        *self.true_params[class_index][i]["var"]])
        self.psi_true = torch.Tensor(data_generation).repeat(2 * class_size, 1).to(self._device)

        #psi_as_dict = {1: self.psi_true[:, :self.n_class_params], 0: self.psi_true[:, self.n_class_params:]}
        psi_as_dict = self.psi_true.reshape(-1, 2, 12).transpose(0, 1)
        self.test_data = self.sample_toy_data_pt(n_classes=2, n_components=3, psi=psi_as_dict).to(self._device)

    def train_discriminator(self):
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 10),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(10, 16),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(16, 1)).to(self._device)

        opt = torch.optim.Adam(self.net.parameters())

        n_epochs = 200
        for e in range(n_epochs):
            output = self.net(self.test_data[:, :-1])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, self.test_data[:, -1].reshape(-1,1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            # with torch.no_grad():
            #     output = net(train_data[:, :-1])
            #     loss = torch.nn.functional.cross_entropy(output, train_data[:, -1].long())
            #     val_metric = accuracy_score(train_data[:, -1].long(), output.argmax(dim=1))
            #     print(loss.item(), val_metric)

        for param in self.net.parameters():
            param.requires_grad_(False)
        print(loss.item())


    def loss(self, y):
        self.net.zero_grad()
        output = self.net(y[:, :-1])
        mask = y[:, -1] > 0.5
        regulariser = y[:, -1][mask].mean()
        lam = 1

        return torch.nn.functional.binary_cross_entropy_with_logits(output,
                                                                    torch.clamp(y[:, -1].reshape(-1,1), 0., 1.),
                                                                    reduction='none') + lam * (regulariser - 1) ** 2

    def sample_toy_data_pt(self, n_classes=2, n_components=3, psi=None):
        means_index = [0, 1, 4, 5, 8, 9]
        std_index = [2, 3, 6, 7, 10, 11]

        n_samples = len(psi[0])
        classes_mask = pyro.sample('class_selection',
                                   dist.Categorical(torch.Tensor([1 / 2, 1 / 2]).view(1, -1).repeat(n_samples, 1)))
        classes_mask = classes_mask.to(self._device)

        data = []
        for class_index in pyro.plate("y", n_classes):
            probs = torch.Tensor([1. / n_components] * n_components).repeat(n_samples, 1)
            assignment = pyro.sample('assignment', dist.Categorical(probs))#.to(self._device)
            means = psi[class_index][:, means_index].reshape(-1, 3, 2).to(torch.device('cpu'))
            stds = psi[class_index][:, std_index].reshape(-1, 3, 2).repeat(1, 1, 2).reshape(-1, 3, 2, 2).to(torch.device('cpu'))
            stds[:, :, 1, :] = stds[:, :, 1, [1, 0]]
            n_dist = dist.MultivariateNormal(means.gather(1, assignment.view(-1, 1).unsqueeze(2).repeat(1, 1, 2)),
                                             stds.gather(1,  assignment.view(-1, 1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 2)))

            data.append(pyro.sample("y_{}".format(class_index), n_dist))
        data = torch.stack(data).to(self._device)
        data = data.gather(0, classes_mask.view(1, -1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 2))[0, :, 0, :]
        data = torch.cat([data, classes_mask.view(-1, 1).float()], dim=1)
        return data

    def _generate_dist(self, psi, x):
        #return self.mixture_model(psi, x)
        raise NotImplementedError

    def _generate(self, psi, x):
        #sigm_ind = list(sum([(i, i + 1) for i in range(2, 24, 4)], ()))
        #psi[:, sigm_ind] = torch.exp(psi[:, sigm_ind])

        # messy stuff to put fixed stds in place
        fixed_std_dim = list(sum([(i, i + 1) for i in range(2, 24, 4)], ()))
        mu_dim = list(sum([(i, i + 1) for i in range(0, 24, 4)], ()))

        output = torch.ones([len(psi), self.psi_true.shape[1]]).to(self._device)
        output[:, mu_dim] = psi
        output[:, fixed_std_dim] = self.psi_true[:1, fixed_std_dim].repeat(len(psi), 1)

        psi_as_dict = output.reshape(-1, 2, 12).transpose(0,1)
        return self.sample_toy_data_pt(psi=psi_as_dict)


class FreqModulatedSoundWave(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=2,
                 loss = lambda y: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss

    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)

class LennardJonesPotentialProblem(YModel):
    def __init__(self, device,
                 psi_init: torch.Tensor,
                 x_range=torch.Tensor(((-2, 0), (2, 5))),
                 x_dim=2, y_dim=2,
                 loss = lambda y: OptLoss.SigmoidLoss(y, 0, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._x_dist = dist.Uniform(*x_range)
        self._psi_dim = len(psi_init)
        self._device = device
        self.loss = loss
    
    @staticmethod
    def g(x):
        return (x[:, 1:] - x[:, :-1].pow(2)).pow(2).sum(dim=1,
                                                        keepdim=True) + (1 - x[:, :-1]).pow(2).sum(dim=1, keepdim=True)


class OptLoss(object):
    def __init__(self):
        pass
    
    @staticmethod
    def R(ys: torch.tensor, Y_0=-5):
        return (ys - Y_0).pow(2).mean(dim=1)

    @staticmethod
    def SigmoidLoss(ys: torch.tensor, left_bound, right_bound):
        return -torch.mean(torch.sigmoid(ys - left_bound) - torch.sigmoid(ys - right_bound), dim=1)

    @staticmethod
    def TanhLoss(ys: torch.tensor, left_bound, right_bound):
        return -torch.mean(torch.tanh(ys - left_bound) - torch.tanh(ys - right_bound), dim=1)


class SHiPModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 device,
                 psi_init: torch.Tensor,
                 address: str = 'http://13.85.29.208:5432',
                 cut_veto=100):
        # super(YModel, self).__init__(y_model=None,
        #                              psi_dim=len(psi_init),
        #                              x_dim=3, y_dim=3) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._psi_dim = len(psi_init)
        self._device = device
        self._cut_veto = cut_veto
        self._address = address

    def sample_x(self, num_repetitions):
        p = np.random.uniform(low=1, high=10, size=num_repetitions)  # energy gen
        phi = np.random.uniform(low=0, high=2 * np.pi, size=num_repetitions)
        theta = np.random.uniform(low=0, high=10 * np.pi / 180)
        pz = p * np.cos(theta)
        px = p * np.sin(theta) * np.sin(phi)
        py = p * np.sin(theta) * np.cos(phi)
        return torch.tensor(np.c_[px, py, pz]).float().to(device)

    @property
    def _y_model(self):
        return self

    @property
    def device(self):
        return self._device

    def _request_data(self, uuid, wait=True):
        r = requests.post("{}/retrieve_result".format(self._address), json={"uuid": uuid})
        r = json.loads(r.content)
        if r["container_status"] == "exited":
            return r
        if wait:
            while r["container_status"] not in ["exited", "failed"]:
                time.sleep(2.)
                r = requests.post("{}/retrieve_result".format(self._address), json={"uuid": uuid})
                r = json.loads(r.content)
            if r["container_status"] == "failed":
                ValueError("Generation has failed!")
            elif r["container_status"] == "exited":
                return r
            return r
        return None

    def _request_uuid(self, condition, num_repetitions):
        x_begin, x_end, y_begin, y_end, z = condition.detach().cpu().numpy()
        r = requests.post(
            "{}/simulate".format(self._address),
            json=json.loads(json.dumps({
                "field": {"Y": 4, "X": 0.0, "Z": 0},
                "shape": {'X_begin': x_begin, "X_end": x_end,
                          'Y_begin': y_begin, "Y_end": y_end, 'Z': z},
                "num_repetitions": num_repetitions
            }, cls=NumpyEncoder))
        )
        return r.content.decode()

    def _loss(self, data):
        data['muons_momentum'] = np.array(data['muons_momentum'])
        data['veto_points'] = np.array(data['veto_points'])
        res = np.sqrt((data['veto_points'][:, :2] ** 2).sum(axis=1)).mean()
        return res

    def _generate(self, condition, num_repetitions):
        uuid = self._request_uuid(condition, num_repetitions=num_repetitions)
        data = self._request_data(uuid, wait=True)
        return data

    def _generate_multiple(self, condition, num_repetitions):
        # making request to calculate new points
        res = {}
        uuids = []
        uuids_to_condition = {}
        for cond in condition:
            uuid = self._request_uuid(cond, num_repetitions=num_repetitions)
            uuids.append(uuid)
            uuids_to_condition[uuid] = cond

        uuids_original = uuids.copy()
        # iterate over uuids
        # and collect computation results from SHiP service
        uuids_processed = []
        while len(uuids):
            time.sleep(2.)
            for uuid in uuids:
                answer = self._request_data(uuid, wait=False)
                if (answer == 'failed') or (answer is not None):
                    uuids_processed.append(uuid)
                    if answer != 'failed':
                        res[uuid] = answer
                        res[uuid]['condition'] = uuids_to_condition[uuid]
                    else:
                        ValueError("Generation has failed for {}!".format(uuids_to_condition[uuid]))
            uuids = list(set(uuids) - set(uuids_processed))
        return uuids_original, res

    def _func(self, condition, num_repetitions):
        res = self._generate(condition, num_repetitions=num_repetitions)
        loss = self._loss(res)
        return loss

    def _func_multiple(self, condition, num_repetitions):
        uuids, data = self._generate_multiple(condition, num_repetitions=num_repetitions)
        loss = []
        for uuid in uuids:
            d = data.get(uuid, None)
            loss.append(self._loss(d))
        return loss

    def generate(self, condition, num_repetitions=100, **kwargs):
        if condition.ndim == 1:
            data = self._generate(condition, num_repetitions=num_repetitions)
            return torch.tensor(data['veto_points']).float().to(condition.device)
        elif condition.ndim == 2:
            uuids, data = self._generate_multiple(condition, num_repetitions=num_repetitions)
            res = np.concatenate([data[uuid]['veto_points'] for uuid in uuids])
            return torch.tensor(res).float().to(device=condition.device)

    def func(self, condition, num_repetitions=100, **kwargs):
        if condition.ndim == 1:
            res = self._func(condition, num_repetitions=num_repetitions)
        elif condition.ndim == 2:
            res = self._func_multiple(condition, num_repetitions=num_repetitions)
        else:
            ValueError('No!')
        return torch.tensor(res).float().to(device=condition.device)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        condition = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)

        condition = step * (condition * 2 - 1) + current_psi
        condition = torch.tensor(condition).float().to(self.device)
        torch.clamp(condition, 1e-5, 1e5)
        uuids, data = self._generate_multiple(condition, num_repetitions=n_samples_per_dim)
        y = []
        xs = []
        psi = []
        for uuid in uuids:
            xs.append(data[uuid]['muons_momentum'])
            y.append(data[uuid]['veto_points'])
            cond = data[uuid]['condition']
            num_entries = len(data[uuid]['muons_momentum'])
            psi.append(cond.repeat(num_entries, 1))
        xs = torch.tensor(np.concatenate(xs)).float().to(self.device)
        y = torch.tensor(np.concatenate(y)).float().to(self.device)
        psi = torch.cat(psi)

        return y, torch.cat([psi, xs], dim=1)

    def loss(self, y, condition):
        pass

    def fit(self, y, condition):
        pass

    def log_density(self, y, condition):
        pass
