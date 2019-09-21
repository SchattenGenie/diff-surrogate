import subprocess
import shlex
import time
import os
import click


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--init_psi', type=str, default="0., 0.")
def main(model, model_config_file, optimized_function, init_psi):
    psi_dim = len([float(x.strip()) for x in init_psi.split(',')])
    lrs = [1e-4, 1e-3, 1e-2, 1e-1]
    n_samples_search = [psi_dim // 2, psi_dim, 2 * psi_dim, 3 * psi_dim]
    command = "python end_to_end.py --model {0} --project_name grid_search \
    --work_space schattengenie --model_config_file {1} --tags {0},{2},grid_search \
    --optimizer TorchOptimizer --optimized_function {2}  --init_psi {3} \
    --n_samples {4} --lr {5} --reuse_optimizer True"
    processes = []
    for lr in lrs:
        for n_samples in n_samples_search:
            command_pre = shlex.split(
                command.format(
                    model,  # 0
                    model_config_file,  # 1
                    optimized_function,  # 2
                    init_psi,  # 3
                    n_samples,  # 4
                    lr,  # 5
                )
            )
            print(command_pre)
            process = subprocess.Popen(command_pre,
                                       shell=False,
                                       close_fds=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       preexec_fn=os.setsid)
            processes.append(process)

    for process in processes:
        print(process.pid)

if __name__ == "__main__":
    main()
