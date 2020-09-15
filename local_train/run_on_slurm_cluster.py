from comet_ml import API
import comet_ml
import io
import os
import numpy as np
from collections import defaultdict
import shlex
import json
import subprocess
import time
import click

def get_NonlinearSubmanifoldHump_experiments(comet_api):
    experiments = comet_api.get(workspace='schattengenie', project_name='gaussianmixturehumpmodeldeepdegenerate')
    return experiments

def get_BostonNN_experiments(comet_api):
    experiments = comet_api.get(workspace='schattengenie', project_name='bostonnntuning')
    return experiments

def get_R10_experiments(comet_api):
    experiments = comet_api.get(workspace='schattengenie', project_name='r10') + comet_api.get(workspace='schattengenie', project_name='r10bock')

    problems = [
        "shir994/rosenbrock/2eaa96e2c2a44d949be827984adcd0bc",  # lts
        "schattengenie/void-baseline/ffa52ac7a8cb45429f5110e13fd5cabf",  # void ffjord
        "schattengenie/numerical-baseline/aefa45d5ea8b492493b1af440d2d2fca",  # numerical diff
        "schattengenie/diff-sim/8a7c0dca061d4753a2c55fd80d6952f5",  # num derivatives
        'schattengenie/cma-es-r/86d3db2fe15c48a881b3bc82b1cd17d9',  # cma es
        'schattengenie/true-grads-r/41f02bab890343f89db4839482b43571'  # true grads
    ]

    experiments = experiments + [
        comet_api.get_experiment(*s.split('/')) for s in problems
    ]
    return experiments

def get_SubmanifoldRosenbrock100_experiments(comet_api):
    experiments = (comet_api.get(workspace='schattengenie', project_name='rosenbrockmodeldegenerate')
                   + comet_api.get(workspace='schattengenie', project_name='rosenbrockmodeldegeneratetest'))

    problems = [
        "schattengenie/rosenbrockmodeldegeneratelts/7dac8f7f0b3440bbac3efdf1be13726d",  # lts
        "schattengenie/rosenbrockmodeldegeneratevoid/50c1739daffb48b4aacc8654ace0ce8c",  # void
        # "schattengenie/diff-sim-baseline/df36eb86b96f45e1990cb1b71280386a", # new num diff
        'schattengenie/rosenbrockmodeldegenerateltscma/020acf906265486ba24596feeb037cc2',  # cma es
        'schattengenie/true-grads-r-i/b3f683cba14646df91a23905078b5c90'  # true grads
    ]

    experiments = experiments + [
        comet_api.get_experiment(*s.split('/')) for s in problems
    ]
    return experiments

def get_Hump_experiments(comet_api):
    problems = [
        "shir994/hump-model/9de86fbf63784d6c85cbd2453222750c",  # lts
        "schattengenie/gp-opt/7b6f49d5838e4dfeb96c2eadf7f986c7",  # gp
        "schattengenie/humpic/8a545ffd6fee42ac90f2ec5346244199",  # numerical diff
        "schattengenie/humpic/b1b81e5d32744ebfa572038bd9e6344f",
        "schattengenie/void-baseline/1562ab81653e4319b362e52a0a362651",  # void
        'schattengenie/cma-es-hump/28cd13aab43e4c68b846df3ed7a6db0e',  # cma es
        'schattengenie/true-grads-hump/5a30c262fd18490eaee4ed229a158784',  # true gradients

        "shir994/hump-model/2daf8455a6f64b698d7f8c8d0f4c3699",
        "shir994/cv-gaussianmixturehumpmodel/7faee83248cd40a291d865cc3ab55326",
        "shir994/cv-gaussianmixturehumpmodel/df9b6c63d47d4725a4d82b7d005ee6d9",
        "shir994/cv-gaussianmixturehumpmodel/c2a8d91214604958ad8acf984f37c40e",

        'schattengenie/cv-gaussianmixturehumpmodel/37fbf6b47e0a4b0b8608519b1e782173',
        'schattengenie/cv-gaussianmixturehumpmodel/f677cf41b2e24cb3b3c73527bfe4a003',
        # 'schattengenie/cv-gaussianmixturehumpmodel/73ec2faf015a4a099a6b773e170cafba',
        'schattengenie/cv-gaussianmixturehumpmodel/d354003fa6c24e4496b7679014e49d79',
        'schattengenie/cv-gaussianmixturehumpmodel/5b12e919c51d40fd8bacfd0653018515',
        'schattengenie/cv-gaussianmixturehumpmodel/55833ec0034c4e35abd7272886f2b320',
    ]

    experiments = [
        comet_api.get_experiment(*s.split('/')) for s in problems
    ]
    try:
        experiments_add = comet_api.get(workspace='schattengenie', project_name='HumpExp')
    except:
        experiments_add = []
    experiments = experiments + experiments_add
    return experiments



def get_parameter_by_key(exp, key):
    parameters = exp.get_parameters_summary()
    for parameter in parameters:
        if parameter['name'] == key:
            return parameter['valueCurrent']
    return None


def new_to_old_metric(exp, key):
    metric = exp.get_metrics(key)
    vals = [float(m['metricValue']) for m in metric]
    return vals


def stack_lists(data, n=1000):
    new_data = []
    for d in data:
        if len(d) > n:
            new_data.append(d[:n])
        elif len(d) < n:
            new_data.append(
                np.concatenate([d, d[-1].repeat(n - len(d))])
            )
    return np.vstack(new_data).T


def preprocess_gp(vals):
    return np.minimum.accumulate(vals)


def add_zero_point(vals, point):
    return np.array([point] + vals.tolist())


def collect_data_from_method(experiments, method: str):
    data = defaultdict(list)
    for experiment in experiments:
        n_samples = get_parameter_by_key(experiment, 'n_samples')
        if method == "GAN":
            if n_samples and (("GAN" in experiment.get_tags()) or ("GANModel" in experiment.get_tags())):
                data[int(n_samples)].append(np.array(new_to_old_metric(experiment, 'Func value')))
            continue

        if method == "FFJORD":
            if n_samples and (("FFJORD" in experiment.get_tags()) or ("FFJORDModel" in experiment.get_tags())):
                data[int(n_samples)].append(np.array(new_to_old_metric(experiment, 'Func value')))
            continue

        if method in experiment.get_tags():
            func_value = np.array(new_to_old_metric(experiment, 'Func value'))
            data[method].append(func_value)
    return data


# setting up constants
methods = [
    "GAN",
    "FFJORD",
    "lts",
    "num_diff",
    "void",
    "gp",
    "cma_es"
]

parameters_per_problem = {
    "Hump": {
        "psi_dim": 2,
        "x_dim": 2,
        "y_dim": 1
    },
    "Rosenbrock10": {
        "psi_dim": 10,
        "x_dim": 1,
        "y_dim": 1
    },
    "SubmanifoldRosenbrock100": {
        "psi_dim": 100,
        "x_dim": 1,
        "y_dim": 1
    },
    "NonlinearSubmanifoldHump": {
        "psi_dim": 40,
        "x_dim": 1,
        "y_dim": 1
    },
    "BostonNN": {
        "psi_dim": 91,
        "x_dim": 13,
        "y_dim": 1
    }
}


@click.command()
@click.option('--problem_to_run', type=str, default='NonlinearSubmanifoldHump')
@click.option('--batch_size', type=int, default=20)
def main(problem_to_run="NonlinearSubmanifoldHump", batch_size=20):
    comet_api = API()
    comet_api.get()

    data = open('./diff_surrogates_commands.txt').read().replace("---", "")
    data = [d for d in data.split("#") if len(d)]
    commands_to_run_per_problem_per_method = defaultdict(dict)
    for problem in data:
        problem_name = problem.split("\n")[0].replace(" ", "")
        print(problem_name)
        commands = [d.replace(u'\xa0', u' ') for d in problem.split("\n")[1:] if (len(d) and not "```" in d)]
        problem_name = problem_name.replace(" ", "")
        for command in commands:
            for method in methods:
                if method in command:
                    if method == "GAN" or method == "FFJORD":
                        if method in command:
                            command_split = shlex.split(command)
                            n_samples = command_split[command_split.index("--n_samples") + 1]
                            commands_to_run_per_problem_per_method[problem_name][n_samples + "_" + method] = command
                            continue
                    commands_to_run_per_problem_per_method[problem_name][method] = command

    # setting up configs
    import gan_config
    with open("gan_config.py", "w") as dict_file:
        gan_config.model_config["psi_dim"] = parameters_per_problem[problem_to_run]["psi_dim"]
        gan_config.model_config["x_dim"] = parameters_per_problem[problem_to_run]["x_dim"]
        gan_config.model_config["y_dim"] = parameters_per_problem[problem_to_run]["y_dim"]
        dict_file.write('model_config = {}'.format(str(gan_config.model_config)))
    import ffjord_config
    with open("ffjord_config.py", "w") as dict_file:
        ffjord_config.model_config["psi_dim"] = parameters_per_problem[problem_to_run]["psi_dim"]
        ffjord_config.model_config["x_dim"] = parameters_per_problem[problem_to_run]["x_dim"]
        ffjord_config.model_config["y_dim"] = parameters_per_problem[problem_to_run]["y_dim"]
        dict_file.write('model_config = {}'.format(str(ffjord_config.model_config)))
    from void import void_config
    with open("./void/void_config.py", "w") as dict_file:
        void_config.model_config["psi_dim"] = parameters_per_problem[problem_to_run]["psi_dim"]
        void_config.model_config["x_dim"] = parameters_per_problem[problem_to_run]["x_dim"]
        void_config.model_config["y_dim"] = parameters_per_problem[problem_to_run]["y_dim"]
        dict_file.write('model_config = {}'.format(str(void_config.model_config)))
    from learn_to_sim import lts_config
    with open("./learn_to_sim/lts_config.py", "w") as dict_file:
        lts_config.model_config["psi_dim"] = parameters_per_problem[problem_to_run]["psi_dim"]
        lts_config.model_config["x_dim"] = parameters_per_problem[problem_to_run]["x_dim"]
        lts_config.model_config["y_dim"] = parameters_per_problem[problem_to_run]["y_dim"]
        dict_file.write('model_config = {}'.format(str(lts_config.model_config)))

    command_to_sh = """#!/bin/bash
set -x
{0}
{1}
"""

    command_cluster = "sbatch -c {0} -t {1} --gpus={2} run_clustering.sh"
    something_to_execute = True
    processes = []
    while something_to_execute:
        something_to_execute = False

        if problem_to_run == "Hump":
            experiments = get_Hump_experiments(comet_api)
        elif problem_to_run == "Rosenbrock10":
            experiments = get_R10_experiments(comet_api)
        elif problem_to_run == "NonlinearSubmanifoldHump":
            experiments = get_NonlinearSubmanifoldHump_experiments(comet_api)
        elif problem_to_run == "BostonNN":
            experiments = get_BostonNN_experiments(comet_api)
        elif problem_to_run == "SubmanifoldRosenbrock100":
            experiments = get_SubmanifoldRosenbrock100_experiments(comet_api)

        # iterating over
        for method in methods:
            d = collect_data_from_method(experiments=experiments, method=method)
            names = [m for m in commands_to_run_per_problem_per_method[problem_to_run].keys() if method in m]
            for name in names:
                command = commands_to_run_per_problem_per_method[problem_to_run][name]
                if method in ["GAN", "FFJORD"]:
                    num_runs = len(d[int(name.split('_')[0])])
                else:
                    num_runs = len(d[method])

                if int(num_runs) < 10:
                    something_to_execute = True
                    if method == "void":
                        command_to_sh_formatted = command_to_sh.format("cd ./void/", command)
                    elif method == "num_diff" or method == "cma_es":
                        command_to_sh_formatted = command_to_sh.format("cd ./baseline_scripts/", command)
                    elif method in ["gp", "GAN", "FFJORD"]:
                        command_to_sh_formatted = command_to_sh.format("cd ./", command)
                else:
                    continue
                print(method, num_runs)
                with open("run_command.sh", "w") as file:
                    file.write(command_to_sh_formatted)
                time.sleep(1)

                if method == "void":
                    command_cluster_formatted = command_cluster.format(2, 24 * 60, 0)  # 24 hours
                elif method == "num_diff" or method == "cma_es":
                    command_cluster_formatted = command_cluster.format(2, 24 * 60, 0)  # 24 hours
                elif method in ["gp"]:
                    command_cluster_formatted = command_cluster.format(2, 5 * 24 * 60, 0)  # 5 days
                elif method in ["GAN", "FFJORD"]:
                    command_cluster_formatted = command_cluster.format(2, 2 * 24 * 60, 1)  # 2 days, 1 GPU

                print(method, name)
                print(command_to_sh_formatted)
                process = subprocess.Popen(
                    command_cluster_formatted,
                    shell=True,
                    close_fds=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                processes.append(process)
                pr_count = subprocess.Popen("squeue | grep vbelavin | wc -l", shell=True, stdout=subprocess.PIPE)
                out, err = pr_count.communicate()
                if int(out) > batch_size:
                    while int(out) > batch_size:
                        print("Waiting... ")
                        time.sleep(60)
                        pr_count = subprocess.Popen("squeue | grep vbelavin | wc -l", shell=True, stdout=subprocess.PIPE)
                        out, err = pr_count.communicate()


if __name__ == "__main__":
    main()
