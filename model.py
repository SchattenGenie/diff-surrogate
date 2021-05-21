import torch
import pyro
import numpy as np
from pyro import distributions as dist
from local_train.base_model import BaseConditionalGenerationOracle
from sklearn.datasets import load_boston
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import norm
from pyro import poutine
import matplotlib.pyplot as plt
import scipy
from pyDOE import lhs
import seaborn as sns
import lhsmdu
import tqdm
import requests
import traceback
import json
import time
import os
import uproot
import sys


from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def volume_of_frastrum(half_h, area_1, area_2):
    return 1 / 3. * 2 * half_h * (area_1 + area_2 + torch.sqrt(area_1 * area_2))


def calculate_section_volume_with_material(z, f_l, f_r, h_l, h_r, g_l, g_r):
    """
    Calculates volume of non-empty pieces of one section of magnet.
    The parameter naming is the same as in Petr Gorbunov ship memo.
    TODO: there is a small descripancy about 5% between this way of calculating
    volume and results from FairShip. Find its cause later.
    """
    area_1 = 2 * (f_l + g_l + f_l) * 2 * (h_l + f_l)
    area_2 = 2 * (f_r + g_r + f_r) * 2 * (h_r + f_r)
    total_volume = volume_of_frastrum(z, area_1, area_2)
    gap_volume = volume_of_frastrum(z, g_l * 2 * h_l, g_r * 2 * h_r)
    return total_volume - 2 * gap_volume


def calculate_weight(volume, density=7.87):
    """
    volume should be in cm^3
    density in g / cm^3
    return mass in kg
    """
    return volume * density / 1e3


def average_block_wise(x, num_repetitions):
    n = x.shape[0]
    if len(x.shape) == 1:
        return F.avg_pool1d(x.view(1, 1, n),
                            kernel_size=num_repetitions,
                            stride=num_repetitions).view(-1)
    elif len(x.shape) == 2:
        cols = x.shape[1]
        return F.avg_pool1d(x.unsqueeze(0).transpose(1, 2),
                            kernel_size=num_repetitions,
                            stride=num_repetitions)[0].transpose(1, 0)
    else:
        NotImplementedError("average_block_wise do not support >2D tensors")


class YModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 device,
                 psi_init: torch.Tensor,
                 x_range: tuple = (-10, 10), y_dim=1,
                 loss=lambda y, **kwargs: OptLoss.SigmoidLoss(y, 5, 10)):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=1, y_dim=y_dim)  # hardcoded values
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

    def generate_data_at_point(self, n_samples_per_dim, current_psi):
        xs = self.sample_x(n_samples_per_dim)
        mus = current_psi.repeat(n_samples_per_dim, 1).clone().detach()
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)

    def generate_local_data(self, n_samples_per_dim, step, current_psi, std=0.1):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))
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
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))

        if n_samples == 0:
            mus = torch.zeros(0, len(current_psi)).float().to(self.device)
        else:
            mus = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)
        mus = step * (mus * 2 - 1) + current_psi
        mus = torch.cat([mus, current_psi.view(1, -1)])
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1)

    def generate_local_data_lhs_normal(self, n_samples_per_dim, sigma, current_psi, n_samples=2):
        xs = self.sample_x(n_samples_per_dim * (n_samples + 1))
        mus = np.append(lhs(len(current_psi), n_samples), np.ones((1, len(current_psi))) / 2., axis=0)
        for i in range(len(current_psi)):
            mus[:, i] = norm(loc=current_psi[i].item(), scale=sigma[i].item()).ppf(
                mus[:, i]
            )
        mus = torch.tensor(mus).float().to(self.device)
        conditions_grid = mus.clone().detach()
        mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
        self.make_condition_sample({'mu': mus, 'x': xs})
        data = self.condition_sample(1).detach().to(self.device)
        r_grid = average_block_wise(self.loss(data), n_samples_per_dim)
        return data.reshape(-1, self._y_dim), torch.cat([mus, xs], dim=1), conditions_grid, r_grid

class SHiPModel(YModel):
    def __init__(self,
                 device,
                 psi_init: torch.Tensor,
                 address: str = 'http://13.85.29.208:5432',
                 cut_veto=100,
                 x_dim=3,
                 y_dim=3):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=x_dim, y_dim=y_dim) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._psi_dim = len(psi_init)
        self._device = device
        self._cut_veto = cut_veto
        self._address = address
        self._left_bound = -300
        self._right_bound = 300
        self.hits_key = "veto_points"
        self.kinematics_key = "muons_momentum"
        self.condition_key = "condition"
        self.saved_muon_input_kinematics = None
        self._iteration_time_limit = 90. # limit of waiting for one iteration in minutes
        self.path_to_enhanced = os.path.expanduser("~/muGAN/SHiP_GAN_module/data_files/" +
                                                   "Seed_auxiliary_values_for_enhanced_generation.npy")
        self.root_filename = "gan_sampled_input.root"
        self.path_to_output_root = os.path.join("/mnt/shipfs/", self.root_filename)
        # self._muGAN = muGAN()

    def sample_x(self, num_repetitions):
        p = np.random.uniform(low=1, high=10, size=num_repetitions)  # energy gen
        phi = np.random.uniform(low=0, high=2 * np.pi, size=num_repetitions)
        theta = np.random.uniform(low=0, high=10 * np.pi / 180)
        pz = p * np.cos(theta)
        px = p * np.sin(theta) * np.sin(phi)
        py = p * np.sin(theta) * np.cos(phi)
        particle_type = np.random.choice([-13., 13.], size=num_repetitions)
        return torch.tensor(np.c_[px, py, pz, particle_type]).float().to(self.device)

    @property
    def _y_model(self):
        return self

    @property
    def device(self):
        return self._device

    def _request_data(self, uuid, wait=True, check_dims=True):
        r = requests.post("{}/retrieve_result".format(self._address), json={"uuid": uuid})
        r = json.loads(r.content)
        if wait:
            while r["container_status"] not in ["exited", "failed"]:
                time.sleep(2.)
                r = requests.post("{}/retrieve_result".format(self._address), json={"uuid": uuid})
                r = json.loads(r.content)
            if r["container_status"] == "failed":
                raise ValueError("Generation has failed with error {}".format(r.get("message", None)))
        if check_dims and r['container_status'] == "exited":
            assert np.array(r[self.condition_key]).shape[0] == self._psi_dim
            assert np.array(r[self.kinematics_key]).shape[1] == self._x_dim
            assert np.array(r[self.hits_key]).shape[1] == self._y_dim
        return r

    def _request_uuid(self, condition, num_repetitions):
        x_begin, x_end, y_begin, y_end, z = torch.clamp(condition, 1e-5, 1e5).detach().cpu().numpy()
        d = {
                "field": {"Y": 4, "X": 0.0, "Z": 0},
                "shape": {'X_begin': x_begin, "X_end": x_end,
                          'Y_begin': y_begin, "Y_end": y_end, 'Z': z},
                "num_repetitions": num_repetitions
            }
        r = requests.post(
            "{}/simulate".format(self._address),
            json=json.loads(json.dumps(d, cls=NumpyEncoder))
        )
        print(r.content, d)
        return r.content.decode()

    def _generate(self, condition, num_repetitions, input_file=None, const_field=None):
        uuid = self._request_uuid(condition, num_repetitions=num_repetitions, input_file=input_file,
                                  const_field=const_field)
        time.sleep(2.)
        data = self._request_data(uuid, wait=True)
        return data

    def _generate_multiple(self, condition, num_repetitions, input_file=None, const_field=True):
        # making request to calculate new points
        res = {}
        uuids = []
        uuids_to_condition = {}
        for cond in condition:
            uuid = self._request_uuid(cond, num_repetitions=num_repetitions, input_file=input_file,
                                      const_field=const_field)
            uuids.append(uuid)
            uuids_to_condition[uuid] = cond

        uuids_original = uuids.copy()
        # iterate over uuids
        # and collect computation results from SHiP service
        uuids_processed = []
        start_time = time.time()
        while len(uuids) and (time.time() - start_time) / 60. < self._iteration_time_limit:
            time.sleep(5.)
            for uuid in uuids:
                answer = self._request_data(uuid, wait=False)
                # TODO: rewrite
                if answer["container_status"] == 'exited':
                    res[uuid] = answer
                    res[uuid][self.condition_key] = uuids_to_condition[uuid]
                    uuids_processed.append(uuid)
                    print("S ", uuid)
                elif answer["container_status"] == 'failed':
                    print("F ", uuid)
                    uuids_processed.append(uuid)
                    # TODO: ignore?
                    # raise ValueError("Generation has failed with error {}".format(r.get("message", None)))
            uuids = list(set(uuids) - set(uuids_processed))
        print("TIME LIMITED: ", len(uuids))
        print("GM", len(res.keys()))
        return uuids_original, res

    def _func(self, condition, num_repetitions, input_file=None, const_field=None):
        res = self._generate(condition, num_repetitions=num_repetitions, input_file=input_file, const_field=const_field)
        y = torch.tensor(np.array(res[self.hits_key])[:, :2])
        loss = self.loss(y, condition)
        return loss

    def _func_multiple(self, condition, num_repetitions, input_file=None):
        uuids, data = self._generate_multiple(condition, num_repetitions=num_repetitions, input_file=input_file)
        loss = []
        for uuid in uuids:
            d = data.get(uuid, None)
            loss.append(self._loss(d, condition))
        return loss

    def generate(self, condition, num_repetitions=100, **kwargs):
        if condition.ndim == 1:
            data = self._generate(condition, num_repetitions=num_repetitions)
            return torch.tensor(data[self.hits_key]).float().to(condition.device)
        elif condition.ndim == 2:
            uuids, data = self._generate_multiple(condition, num_repetitions=num_repetitions)
            res = np.concatenate([data[uuid][self.hits_key] for uuid in uuids])
            return torch.tensor(res).float().to(device=condition.device)

    def func(self, condition, num_repetitions=100, **kwargs):
        if condition.ndim == 1:
            res = self._func(condition, num_repetitions=num_repetitions, input_file=kwargs["input_file"],
                             const_field=kwargs["const_field"])
        elif condition.ndim == 2:
            res = self._func_multiple(condition, num_repetitions=num_repetitions, input_file=kwargs["input_file"])
        else:
            ValueError('No!')
        return torch.tensor(res).float().to(device=condition.device)

    def generate_local_data_lhs(self, n_samples_per_dim, step, current_psi, n_samples=2):
        condition = torch.tensor(lhs(len(current_psi), n_samples)).float().to(self.device)

        condition = step * (condition * 2 - 1) + current_psi
        condition = torch.tensor(condition).float().to(self.device)
        condition = torch.clamp(condition, 1e-5, 1e5)

        self.sample_from_gan(n_samples_per_dim + 50_000, output_path=self.path_to_output_root)
        uuids, data = self._generate_multiple(condition,
                                              num_repetitions=n_samples_per_dim,
                                              input_file=self.root_filename,
                                              const_field=True)
        print("ORIG ", len(uuids))
        y = []
        xs = []
        psi = []
        for uuid in uuids:
            try:
                print(data[uuid].keys())
            except KeyError as e:
                print(e)
                continue
            num_entries = len(data[uuid][self.kinematics_key])
            if num_entries == 0:
                continue
            xs.append(data[uuid][self.kinematics_key])
            y.append(data[uuid][self.hits_key])
            cond = data[uuid][self.condition_key]
            psi.append(cond.repeat(num_entries, 1))
        if len(xs) == len(y) == 0:
            return None, None
        # TODO: fix in case of 0 entries
        # if there is absolutelt no entires have passed
        xs = torch.tensor(np.concatenate(xs)).float().to(self.device)
        # self.saved_muon_input_kinematics = xs
        y = torch.tensor(np.concatenate(y)).float().to(self.device)
        psi = torch.cat(psi)

        return y[:, :2], torch.cat([psi, xs], dim=1)

    def loss(self, y, conditions):
        if len(conditions.size()) == 1:
            condition = conditions[:self._psi_dim]
        else:
            condition = conditions[0, :self._psi_dim]
        hit_loss = torch.prod(torch.sigmoid(y - self._left_bound) - torch.sigmoid(y - self._right_bound),
                              dim=1).mean(dim=1)

        x_begin, x_end, y_begin, y_end, z = torch.clamp(condition, 1e-5, 1e5).detach().cpu().numpy()

        volume_of_magnet = 1 / 3. * z * (x_begin * y_begin +
                                         x_end * y_end +
                                         torch.sqrt(x_begin * x_end * y_begin * y_end))
        length_reg = 1
        mass_reg = 1
        steel_rho = 8  # kg / m^3

        normalising_constant_mass = 90
        normalising_constant_length = 8
        return hit_loss + \
               length_reg * z / normalising_constant_length + \
               mass_reg * volume_of_magnet * steel_rho / normalising_constant_mass

    def fit(self, y, condition):
        pass

    def log_density(self, y, condition):
        pass

    def grad(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        return torch.zeros_like(condition)

    def sample_from_gan(self, num_repetitions, output_path=None):
        # seed_auxiliary_distributions = np.load(self.path_to_enhanced)
        # seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,
        #                                        np.random.permutation(seed_auxiliary_distributions.shape[0]), axis=0,
        #                                        out=seed_auxiliary_distributions)
        # fraction_to_boost = 0.11
        # cut = int(np.shape(seed_auxiliary_distributions)[0] * fraction_to_boost)
        # dist = np.abs(np.random.normal(loc=0, scale=1, size=np.shape(seed_auxiliary_distributions[:cut, 2])))
        # dist = np.abs(np.random.normal(loc=0, scale=1, size=np.shape(dist)))
        # dist += 1
        # dist = np.power(dist, 0.55)
        # seed_auxiliary_distributions[:cut, 2] *= dist
        # seed_auxiliary_distributions = np.take(seed_auxiliary_distributions,
        #                                        np.random.permutation(seed_auxiliary_distributions.shape[0]), axis=0,
        #                                        out=seed_auxiliary_distributions)

        # boosted_muon_kinematic_vectors = self._muGAN.generate_enhanced(auxiliary_distributions=seed_auxiliary_distributions,
        #                                                          size=num_repetitions)

        # This is super dirty quickfix to generate muons on the fly with TF to get rid of mixing TF and PT being used
        # at the same time. Award of the best coding practices.
        os.system("python ../generate_gan_muons.py {} {}".format(self.path_to_enhanced, num_repetitions))
        print("Waiting until muon generation is done")
        while not os.path.exists("../done.txt"):
            time.sleep(2)
        print("remove flag file")
        os.remove("../done.txt")
        boosted_muon_kinematic_vectors = np.load("../gan_muons.npy")

        if output_path:
            self.save_to_ROOT(data=boosted_muon_kinematic_vectors, filename=output_path)
        else:
            xs = np.empty_like(boosted_muon_kinematic_vectors)
            xs[:, :3] = boosted_muon_kinematic_vectors[:, -3:]
            xs[:, 3:6] = boosted_muon_kinematic_vectors[:, 1:4]
            xs[:, -1] = boosted_muon_kinematic_vectors[:, 0]
            return torch.FloatTensor(xs).to(self.device)


    def save_to_ROOT(self, data, filename='muons.root'):
        '''  Use uproot to save a generated array to a ROOT file that is compalible with MuonBackGenerator.cxx from FairShip'''

        shape = np.shape(data)[0]

        data[:,
        3] += 2084.5  # Shift target to 50m. In accordance with primGen.SetTarget(ship_geo.target.z0+50*u.m,0.) in run_simScript.py
        # The start of target in the GAN training data is -7084.5.

        dtype = '>f4'

        Event_ID = uproot.newbranch(dtype)
        ID = uproot.newbranch(dtype)
        Parent_ID = uproot.newbranch(dtype)
        Pythia_ID = uproot.newbranch(dtype)
        ECut = uproot.newbranch(dtype)
        W = uproot.newbranch(dtype)
        X = uproot.newbranch(dtype)
        Y = uproot.newbranch(dtype)
        Z = uproot.newbranch(dtype)
        PX = uproot.newbranch(dtype)
        PY = uproot.newbranch(dtype)
        PZ = uproot.newbranch(dtype)
        Release_Time = uproot.newbranch(dtype)
        Mother_ID = uproot.newbranch(dtype)
        Process_ID = uproot.newbranch(dtype)

        branchdict = {"event_id": Event_ID, "id": ID, "parentid": Parent_ID, "pythiaid": Pythia_ID, "ecut": ECut, "w": W,
                      "x": X, "y": Y, "z": Z, "px": PX, "py": PY, "pz": PZ, "release_time": Release_Time,
                      "mother_id": Mother_ID, "process_id": Process_ID}

        tree = uproot.newtree(branchdict, title="pythia8-Geant4")

        with uproot.recreate(filename) as f:
            f["pythia8-Geant4"] = tree

            f["pythia8-Geant4"].extend({"event_id": np.ones(shape).astype(np.float64), "id": data[:, 0].astype(np.float64),
                                        "parentid": np.zeros(shape).astype(np.float64),
                                        "pythiaid": data[:, 0].astype(np.float64),
                                        "ecut": np.array(np.ones(shape) * 0.00001).astype(np.float64),
                                        "w": np.ones(shape).astype(np.float64),
                                        "x": np.array(data[:, 1] * 0.01).astype(np.float64),
                                        "y": np.array(data[:, 2] * 0.01).astype(np.float64),
                                        "z": np.array(data[:, 3] * 0.01).astype(np.float64),
                                        "px": data[:, 4].astype(np.float64), "py": data[:, 5].astype(np.float64),
                                        "pz": data[:, 6].astype(np.float64),
                                        "release_time": np.zeros(shape).astype(np.float64),
                                        "mother_id": np.array(np.ones(shape) * 99).astype(np.float64),
                                        "process_id": np.array(np.ones(shape) * 99).astype(np.float64)})
            # Not clear if all the datatype formatting is needed. Can be fiddly with ROOT datatypes. This works so I left it.

            # Add buffer event at the end. This will not be read into simulation.
            f["pythia8-Geant4"].extend({"event_id": [0], "id": [0], "parentid": [0],
                                        "pythiaid": [0], "ecut": [0], "w": [0], "x": [0],
                                        "y": [0], "z": [0], "px": [0], "py": [0],
                                        "pz": [0], "release_time": [0], "mother_id": [0], "process_id": [0]})

        print(' ')
        print(' ')
        print('Saved', shape, 'muons to', filename, '.')
        print('run_simScript.py must be run with the option: -n', shape, '(or lower)')
        print(' ')
        print(' ')

class FullSHiPModel(SHiPModel):
    def __init__(self,
                 device,
                 psi_init: torch.Tensor,
                 address: str = 'http://127.0.0.1:5444',
                 x_dim=7,
                 y_dim=2):
        super().__init__(device=device, psi_init=psi_init,
                         address=address, x_dim=x_dim, y_dim=y_dim)
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._psi_dim = len(psi_init)
        self.hits_key = "veto_points"
        self.kinematics_key = "kinematics"
        self.condition_key = "params"
        self.scale_psi = False
        self.params_precision = 1

    def sample_x(self, num_repetitions):
        # # TODO: For now use boostrap, once
        # # things are working, sample from distributions somehow
        # sample_indices = np.random.choice(
        #     range(len(self.saved_muon_input_kinematics)),
        #     num_repetitions,
        #     replace=True
        # )
        # return self.saved_muon_input_kinematics[sample_indices]

        # Using muGAN to generate samples
        return self.sample_from_gan(num_repetitions, output_path=None)

    def _request_uuid(self, condition, num_repetitions, input_file, const_field):
        d = {"shape": list(map(lambda x: round(x, self.params_precision), condition.detach().cpu().numpy().tolist())),
             "n_events": num_repetitions,
             "n_jobs": 16,
             "input_file": input_file,
             "const_field": const_field}
        print("request_params", d)
        r = requests.post(
            "{}/simulate".format(self._address),
            json=json.loads(json.dumps(d))
        )
        print("content", r.content)
        return r.content.decode()

    def request_params(self, condition):
        d = {"shape": list(map(lambda x: round(x, self.params_precision), condition.detach().cpu().numpy().tolist()))}
        print("request_params", d)
        r = requests.post(
            "{}/retrieve_params".format(self._address),
            json=json.loads(json.dumps(d))
        )
        print("content", r.content)
        return json.loads(r.content)

    def loss(self, y, conditions):
        """
        Vectorised oss function as in Oliver's code
        :param y: 2D distribution of hits
        :param conditions: full matrix of conditions(magenet and kinematic)
        :return:
        """

        ## NOTE! In reality muon is -13, but because intially I copied loss function definition from Thesis, mu+
        ## and mu- are actually flipped through the whole loss.
        MUON = 13
        left_margin = 3.0  # in m
        right_margin = 3  # in m
        margin_sum = left_margin + right_margin
        y_margin = 5  # in m
        y = y / 100. # convert cm to m
        print("inside loss kinematics example value: {}".format(conditions[0, self._psi_dim:]))

        acceptance_mask_plus = (y[:, 0] <= left_margin) & (-right_margin <= y[:, 0]) & (torch.abs(y[:, 1]) < y_margin) & (conditions[:, -1] == MUON)
        acceptance_mask_minus = (y[:, 0] <= right_margin) & (-left_margin <= y[:, 0]) & (torch.abs(y[:, 1]) < y_margin) & (conditions[:, -1] == -MUON)

        print((acceptance_mask_plus & acceptance_mask_minus).sum())
        # 1e-5 and .abs() to prevent bad gradients of sqrt(-0), which leads to NaN in .grad for psi
        sum_term_1 = (acceptance_mask_plus.float()) * torch.sqrt(1e-5 + ((margin_sum - (y[:, 0] + right_margin)) / margin_sum).abs())
        # get rid of NaN
        sum_term_1[sum_term_1 != sum_term_1] = 0.
        sum_term_2 = (acceptance_mask_minus.float()) * torch.sqrt(1e-5 + ((margin_sum + (y[:, 0] - right_margin)) / margin_sum).abs())
        sum_term_2[sum_term_2 != sum_term_2] = 0.

        sum_term = sum_term_1 + sum_term_2

        W_star = torch.tensor(1915820.).to(self._device)
        # TODO: Dont want to run a k8s job for just parameter check for now
        W_ship = None  # torch.tensor(self.request_params(conditions[0, :self._psi_dim])["w"]).to(self._device)
        if self.scale_psi:
            W = self.calculate_weight(conditions[0, :self._psi_dim] / self.scale_factor * self.feature_max)
        else:
            W = self.calculate_weight(conditions[0, :self._psi_dim])
        print("Analytic mass: {}, true mass {}".format(W, W_ship))

        # weight_loss = 1 + torch.exp(10. * (W - W_star) / W_star)
        # hits_loss = 1. + sum_term
        weight_loss = torch.exp(10. * (W - W_star) / W_star)

        hits_loss = sum_term * len(sum_term)
        reg_coeff = 5.

        print("Weight loss: {}, Hits loss: mean {}, sum {}".format(weight_loss, hits_loss.mean(), hits_loss.sum()))

        #  return weight_loss * hits_loss + torch.nn.functional.relu(W - 3e6) * 1e8
        return hits_loss  #  + weight_loss * reg_coeff + torch.nn.functional.relu(W - 3e6) * 1e8

    def _func(self, condition, num_repetitions, input_file=None, const_field=None):
        res = self._generate(condition, num_repetitions=num_repetitions, input_file=input_file, const_field=const_field)
        y = torch.tensor(res[self.hits_key])[:, :2].float().to(self._device)
        xs = torch.tensor(res[self.kinematics_key]).float().to(self._device)
        psi = np.array(res[self.condition_key])
        num_entries = len(xs)
        psi = psi.reshape(1, self._psi_dim).repeat(num_entries, 0)
        psi = torch.tensor(psi).float().to(self._device)
        # TODO: fix in case of 0 entries
        conditions = torch.cat([psi, xs], dim=1)
        loss = self.loss(y, conditions)
        return loss

    def _func_multiple(self, condition, num_repetitions):
        uuids, data = self._generate_multiple(condition, num_repetitions=num_repetitions)
        print("ORIG ", len(uuids))
        y = []
        xs = []
        psi = []
        for uuid in uuids:
            try:
                print(data[uuid].keys())
            except KeyError as e:
                print(e)
                continue
            num_entries = len(data[uuid][self.kinematics_key])
            if num_entries == 0:
                continue
            xs.append(data[uuid][self.kinematics_key])
            y.append(data[uuid][self.hits_key])
            psi.append(data[uuid][self.condition_key])
        if len(xs) == len(y) == 0:
            return None, None
        # TODO: fix in case of 0 entries
        # if there is absolutelt no entires have passed
        losses = []
        for index in range(len(y)):
            _xs = torch.tensor(xs[index]).float().to(self._device)
            _y = torch.tensor(y[index]).float().to(self._device)
            _psi = torch.tensor(psi[index].repeat(len(_xs), 1)).float().to(self._device)
            losses.append(self.loss(_y, torch.cat([_psi, _xs], dim=1)))
        return losses

    def calculate_weight(self, magnet_params):
        magnet_sections = 6
        params_per_section = 6
        total_mass = torch.tensor([0.]).to(self._device)
        for i in range(magnet_sections):
            volume = calculate_section_volume_with_material(magnet_params[i],
                                                            *magnet_params[i * params_per_section + magnet_sections:
                                                                           i * params_per_section +
                                                                           magnet_sections + params_per_section])
            total_mass += calculate_weight(volume)

        magnet_7_volume = calculate_section_volume_with_material(10, magnet_params[-5], magnet_params[-5],
                                               magnet_params[-3], magnet_params[-3],
                                               magnet_params[-1], magnet_params[-1])
        total_mass += calculate_weight(magnet_7_volume)

        # I add mass of absorber (fixed) to make relu activation in loss term acting correctly
        mass_of_absorber = 257401. #  kg
        return total_mass + mass_of_absorber


    # def generate_data_at_point(self, n_samples_per_dim, current_psi):
    #     data = self._generate(current_psi, n_samples_per_dim)
    #     xs = torch.tensor(data[self.kinematics_key]).float().to(self._device)
    #     y = torch.tensor(data[self.hits_key]).float().to(self._device)
    #     cond = data[self.condition_key]
    #     psi = torch.tensor(cond.repeat(n_samples_per_dim, 1)).float().to(self._device)
    #     return y[:, :2], torch.cat([psi, xs], dim=1)


class SimpleSHiPModel(YModel):
    def __init__(self,
                 device,
                 psi_init: torch.Tensor,
                 address: str = 'http://13.85.29.208:5432',
                 cut_veto=100):
        super(YModel, self).__init__(y_model=None,
                                     psi_dim=len(psi_init),
                                     x_dim=3, y_dim=3) # hardcoded values
        self._psi_dist = dist.Delta(psi_init.to(device))
        self._psi_dim = len(psi_init)
        self._device = device
        self._cut_veto = cut_veto
        self._address = address
        self._left_bound = -300
        self._right_bound = 300

    def sample_x(self, num_repetitions):
        p = np.random.uniform(low=1, high=10, size=num_repetitions)  # energy gen
        phi = np.random.uniform(low=0, high=2 * np.pi, size=num_repetitions)
        theta = np.random.uniform(low=0, high=10 * np.pi / 180)
        pz = p * np.cos(theta)
        px = p * np.sin(theta) * np.sin(phi)
        py = p * np.sin(theta) * np.cos(phi)
        particle_type = np.random.choice([-13., 13.], size=num_repetitions)
        return torch.tensor(np.c_[px, py, pz, particle_type]).float().to(self.device)

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
                raise ValueError("Generation has failed with error {}".format(r.get("message", None)))
            elif r["container_status"] == "exited":
                return r
            return r
        return r

    def _request_uuid(self, condition, num_repetitions):
        x_begin, x_end, y_begin, y_end, z = torch.clamp(condition, 1e-5, 1e5).detach().cpu().numpy()
        d = {
                "field": {"Y": 4, "X": 0.0, "Z": 0},
                "shape": {'X_begin': x_begin, "X_end": x_end,
                          'Y_begin': y_begin, "Y_end": y_end, 'Z': z},
                "num_repetitions": num_repetitions
            }
        r = requests.post(
            "{}/simulate".format(self._address),
            json=json.loads(json.dumps(d, cls=NumpyEncoder))
        )
        print(r.content, d)
        return r.content.decode()

    def _loss(self, data):
        data['muons_momentum'] = np.array(data['muons_momentum'])
        data['veto_points'] = np.array(data['veto_points'])
        y = torch.tensor(data['veto_points'][:, :2])
        return torch.prod(torch.sigmoid(y - self._left_bound) - torch.sigmoid(y - self._right_bound), dim=1).mean()

    def _generate(self, condition, num_repetitions):
        # TODO: duct tape
        data = None
        for _ in range(3):
            try:
                uuid = self._request_uuid(condition, num_repetitions=num_repetitions)
                time.sleep(2.)
                data = self._request_data(uuid, wait=True)
                break
            except Exception as e:
                print(e, traceback.format_exc())
        if (data is None) or (len(data['muons_momentum']) == 0):
            data = {
                'muons_momentum': np.zeros(num_repetitions, 4),
                'veto_points': 1000000 * np.ones(num_repetitions, 2)
            }
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
            time.sleep(5.)
            for uuid in uuids:
                answer = self._request_data(uuid, wait=False)
                # TODO: rewrite
                if answer["container_status"] == 'exited':
                    uuids_processed.append(uuid)
                    res[uuid] = answer
                    res[uuid]['condition'] = uuids_to_condition[uuid]
                elif answer["container_status"] == 'failed':
                    uuids_processed.append(uuid)
                    # TODO: ignore? duck tape
                    # raise ValueError("Generation has failed with error {}".format(r.get("message", None)))
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
        condition = torch.clamp(condition, 1e-5, 1e5)
        uuids, data = self._generate_multiple(condition, num_repetitions=n_samples_per_dim)
        y = []
        xs = []
        psi = []
        for uuid in uuids:
            if len(data[uuid]['muons_momentum']):
                xs.append(data[uuid]['muons_momentum'])
                y.append(data[uuid]['veto_points'])
                cond = data[uuid]['condition']
                num_entries = len(data[uuid]['muons_momentum'])
                psi.append(cond.repeat(num_entries, 1))
        try:
            xs = torch.tensor(np.concatenate(xs)).float().to(self.device)
            y = torch.tensor(np.concatenate(y)).float().to(self.device)
            psi = torch.cat(psi)
        except:
            # TODO: even more duck tape
            psi = current_psi.repeat(n_samples_per_dim, 1).float().to(self.device)
            y = (10000 * torch.ones(2)).repeat(n_samples_per_dim, 1).float().to(self.device)
            xs = (torch.zeros(3)).repeat(n_samples_per_dim, 1).float().to(self.device)
        return y[:, :2], torch.cat([psi, xs], dim=1)

    def loss(self, y):
        return torch.prod(torch.sigmoid(y - self._left_bound) - torch.sigmoid(y - self._right_bound), dim=1)

    def fit(self, y, condition):
        pass

    def log_density(self, y, condition):
        pass

    def grad(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        return torch.zeros_like(condition)
