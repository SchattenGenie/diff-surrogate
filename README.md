# diff-surrogate
- [ ] Add full example on how to run a code end to end in the README:
  - [ ] Add the run command for all the experiments we have run, and table of initial opt/GAN params.
- [x] Add back anonymised SHiP model.
- [] ?


## Run instructions (working with GAN):
- create an environment with conda yml file: ```conda env create -f conda_env.yml```

- ```cd /diff-surrogate/local_train```. 
- Execute the commands below for the particular experiment.

In case you also want to check FFJORD you need to install

```pip install git+https://github.com/rtqichen/torchdiffeq```

## GAN surrogate parameters
| Simulator Model | task | y_dim | x_dim | noise_dim | lr | batch_size | epochs | iters_discriminator | iters_generator | instance_noise_std | dis_output_dim | grad_penalty | gp_reg_coeff |
|:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |
| Three Hump Model |x|x|x|x|x|x|x|x|x|x|x|x|x|x|
| Rosenbrock 10dim  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|
| Rosenbrock submanifold 100dim |x|x|x|x|x|x|x|x|x|x|x|x|x|x|
| Three Hump submanifold 40dim  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|
| Neural network optimisation |x|x|x|x|x|x|x|x|x|x|x|x|x|x|

## Optimizer parameters parameters
| Simulator Model | task | torch_model | num_repetitions
|:---:         |     :---:      |          :---: |:---:         |
| Three Hump Model |x|x|x|
| Rosenbrock 10dim  |x|x|x|
| Rosenbrock submanifold 100dim |x|x|x|
| Three Hump submanifold 40dim  |x|x|x|
| Neural network optimisation |x|x|x|



## To reproduce experiments
Dont forget to set the parameters of the used optimisers to the values, presented in the tables above.
(GAN, FFJORD, Void and LTS models).

- Three Hump Model
  - GAN
  
  ```python end_to_end.py --model GANModel --project_name HumpExp --work_space USERNAME --tags GaussianMixtureHumpModel,GANModel --optimizer TorchOptimizer --optimized_function GaussianMixtureHumpModel --step_data_gen 0.5 --init_psi 2.,0. --reuse_optimizer True --n_samples 4 --lr 0.1```
  - FFJORD
  
  ```python end_to_end.py --model FFJORDModel --project_name HumpExp --work_space USERNAME --model_config_file ffjord_config --tags GaussianMixtureHumpModel,FFJORDModel --optimizer TorchOptimizer --optimized_function GaussianMixtureHumpModel --step_data_gen 0.5 --init_psi 2.,0. --n_samples 4 --lr 0.1```
  - Numerical differentiation
  
  ```cd baseline_scripts && python baseline.py --project_name HumpExp --work_space USERNAME --optimized_function GaussianMixtureHumpModel --optimizer TorchOptimizer --tags num_diff,HumpExp --optimizer_config_file optimizer_config_num_diff  --init_psi 2.,0. --h 0.1```
  - Bayesin optimisation (BOCK)
  
  ```cd baseline_scripts && python baseline.py --project_name HumpExp --optimized_function GaussianMixtureHumpModel --work_space USERNAME --optimizer BOCKOptimizer --tags gp,GaussianMixtureHumpModel --optimizer_config_file optimizer_config_gp   --init_psi 2.,0.```
  - Guided evolutionary stratagies
  
  ```cd baseline_scripts && python lts_run.py --model LearnToSimModel --optimizer TorchOptimizer --optimized_function GaussianMixtureHumpModel --model_config_file lts_config --project_name HumpExp --work_space USERNAME --tags lts,GaussianMixtureHumpModel --epochs 2000 --n_samples 25 --n_samples_per_dim 3000 --init_psi 2.,0. --reuse_optimizer True```
  - Void 
  
  ```cd void && python void_run.py --project_name HumpExp --work_space USERNAME --optimizer_config_file optimizer_config_void --tags void,GaussianMixtureHumpModel --optimized_function GaussianMixtureHumpModel  --init_psi 2.,0.```
  - Learning to Simulate
  
  ```python lts_run.py --model LearnToSimModel --optimizer TorchOptimizer --optimized_function GaussianMixtureHumpModel --model_config_file lts_config --project_name HumpExp --work_space USERNAME --tags lts,GaussianMixtureHumpModel --epochs 2000 --n_samples 25 --n_samples_per_dim 3000 --init_psi 2.,0. --reuse_optimizer True```
  
- Rosenbrock 10dim
```code```
- Rosenbrock submanifold 100dim
```code```
- Three Hump submanifold 40dim
```code```
- Neural network optimisation
```code```
