import subprocess
import time
import signal
import os
import comet_ml
import numpy as np
import click
import psutil
import logging
from generate_initial_psi import CreateDiscreteSpace

logging.basicConfig(filename='logging_global.log',level=logging.INFO, filemode='w')

parameters = {"tags": "test_restarts,muGAN",
              "epochs": 100,
              "n_samples": 84,
              "n_samples_per_dim": 500000,
              "step_data_gen": 10,
              "lr": 0.05}

def terminate(proc_pid, timeout=60):
    def on_terminate(proc):
        logging.info("process {} terminated with exit code {}".format(proc, proc.returncode))

    process = psutil.Process(proc_pid)
    children = process.children(recursive=True)
    for proc in children:
        proc.terminate()
    process.terminate()

    gone, alive = psutil.wait_procs(children, timeout=timeout, callback=on_terminate)
    for p in alive:
        p.kill()
    gone, alive = psutil.wait_procs([process], timeout=timeout, callback=on_terminate)
    for p in alive:
        p.kill()


class EarlyStoppping():
    def __init__(self, threshold, fast_descent_patience, patience):
        self.threshold = threshold
        self.patience = patience
        self.fast_descent_patience = fast_descent_patience
        logging.info(threshold, patience, fast_descent_patience)
        self.best_iter = 0
        self.best_score = None

    def need_restart(self, func_values):
        if len(func_values) == 0:
            return False

        if self.best_score is None or func_values[-1] < self.best_score:
            self.best_score = func_values[-1]
            self.best_iter = len(func_values)

        if len(func_values) - self.best_iter > self.patience:
            return True

        if len(func_values) >= self.fast_descent_patience and np.min(func_values) > self.threshold:
            return True
        return False



@click.command()
@click.option('--patience', type=int, default=10)
@click.option('--loss_threshold', type=int, default=1000)
def run_training(patience, loss_threshold):
    search_space = CreateDiscreteSpace()
    restart = True
    early_stopping = EarlyStoppping(loss_threshold, patience // 2, patience)
    while True:
        if restart:
            init_psi = ",".join(list(map(lambda x: str(float(x)), search_space.rvs()[0])))
            command = " ".join(("python end_to_end.py --model GANModel",
                                "--optimizer TorchOptimizer",
                                "--optimized_function FullSHiPModel",
                                "--project_name full_ship",
                                "--work_space shir994",
                                "--tags {}".format(parameters["tags"]),
                                "--epochs {}".format(parameters["epochs"]),
                                "--n_samples {}".format(parameters["n_samples"]),
                                "--n_samples_per_dim {}".format(parameters["n_samples_per_dim"]),
                                "--init_psi {}".format(init_psi),
                                "--reuse_optimizer True",
                                "--step_data_gen {}".format(parameters["step_data_gen"]),
                                "--lr {}".format(parameters["lr"])))

            process = subprocess.Popen(f'bash -c "source ~/data/anaconda3/etc/profile.d/conda.sh;\
                                               conda activate lgso; {command}"', shell=True)
            time.sleep(60 * 5)
            with open("experiment_id.txt", "r") as f:
                experiment_id = f.readline().strip()
            logging.info("Starting experiment: {}".format(experiment_id))

        time.sleep(60 * 5)
        with open("experiment_id.txt", "r") as f:
            experiment_id = f.readline().strip()
        comet_api = comet_ml.API()
        experiment = comet_api.get(experiment_id)
        func_value = np.array([float(metric["metricValue"]) for metric in experiment.get_metrics("Func value")])
        restart = early_stopping.need_restart(func_value)
        if restart:
            logging.info(func_value, restart, len(func_value))
            logging.info("Func len: {}".format(len(func_value)))
            logging.info("Killing experiment: {}".format(experiment_id))
            terminate(process.pid)

if __name__ == "__main__":
    run_training()
