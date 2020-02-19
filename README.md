# diff-surrogate

## Run instructions (working with GAN):
- create an environment with conda yml file: ```conda env create -f conda_env.yml```

- ```cd /diff-surrogate/local_train```. 
- Execute the commands below for the particular experiment.

In case you also want to check FFJORD you need to install

```pip install git+https://github.com/rtqichen/torchdiffeq```

## GAN surrogate parameters
| Simulator Model | task | psi_dim| y_dim | x_dim | noise_dim | lr | batch_size | epochs | iters_discriminator | iters_generator | instance_noise_std | dis_output_dim | grad_penalty | gp_reg_coeff |
|:---:         |:---:         |    :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |
| Three Hump Model |CRAMER|2|1|2|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Rosenbrock 10dim  |CRAMER|10|1|1|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Rosenbrock submanifold 100dim |CRAMER|100|1|1|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Three Hump submanifold 40dim  |CRAMER|40|1|1|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Neural network optimisation |CRAMER|91|1|13|150|8e-4|512|15|1|1|0.01|256|True|10|x|


## Void baseline parameters
`psi_dim, y_dim, x_dim` are set as above. The rest left unchanged.

## Learning to simulate (LTS) baseline parameters
`y_dim, x_dim` are set as above. The rest left unchanged.

## Optimizer parameters parameters
| Simulator Model | lr | torch_model | num_repetitions
|:---:         |     :---:      |          :---: |:---:         |
| Three Hump Model |0.1|Adam|10000|
| Rosenbrock 10dim  |0.1|Adam|10000|
| Rosenbrock submanifold 100dim |0.1|Adam|10000|
| Three Hump submanifold 40dim  |0.1|Adam|10000|
| Neural network optimisation |0.1|Adam|10000|



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
  
  ```cd learn_to_sim && python lts_run.py --model LearnToSimModel --optimizer TorchOptimizer --optimized_function GaussianMixtureHumpModel --model_config_file lts_config --project_name HumpExp --work_space USERNAME --tags lts,GaussianMixtureHumpModel --epochs 2000 --n_samples 25 --n_samples_per_dim 3000 --init_psi 2.,0. --reuse_optimizer True```
  
- Rosenbrock 10dim
  - GAN
  
  ```python end_to_end.py --model GANModel --project_name R10 --work_space USERNAME --tags R10,GANModel --optimizer TorchOptimizer --optimized_function RosenbrockModel --step_data_gen 0.2 --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2. --n_samples 15 --optimizer_config_file optimizer_config --lr 0.1```
  - FFJORD
  
  ```python end_to_end.py --model FFJORDModel --project_name R10 --work_space USERNAME --tags R10,FFJORDModel --model_config_file ffjord_config --optimizer TorchOptimizer --optimized_function RosenbrockModel --step_data_gen 0.2 --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2. --n_samples 15 --optimizer_config_file optimizer_config --lr 0.1```
  - Numerical differentiation
  
  ```cd baseline_scripts && python baseline.py --project_name R10 --work_space USERNAME --optimized_function RosenbrockModel --optimizer TorchOptimizer --tags num_diff,R10 --optimizer_config_file optimizer_config_num_diff  --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2. --h 0.1```
  - Bayesin optimisation (BOCK)
  
  ```cd baseline_scripts && python baseline.py --project_name R10 --optimized_function RosenbrockModel --work_space USERNAME --optimizer BOCKOptimizer --tags gp,GaussianMixtureHumpModel --optimizer_config_file optimizer_config_gp   --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2.```
  - Guided evolutionary stratagies
  
  ```cd baseline_scripts && python baseline.py --project_name R10 --work_space USERNAME --optimized_function RosenbrockModel --optimizer CMAGES --tags cma_es,GaussianMixtureHumpModel --optimizer_config_file optimizer_config_cma_es --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2. --p 10```
  - Void 
  
  ```cd void && python void_run.py --project_name R10 --work_space USERNAME --optimizer_config_file optimizer_config_void --tags void,RosenbrockModel --optimized_function RosenbrockModel  --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2.```
  - Learning to Simulate
  
  ```cd learn_to_sim && python lts_run.py --model LearnToSimModel --optimizer TorchOptimizer --optimized_function RosenbrockModel --model_config_file lts_config --project_name R10 --work_space USERNAME --tags lts,GaussianMixtureHumpModel --epochs 2000 --n_samples 25 --n_samples_per_dim 3000 --init_psi 2.,2.,2.,2.,2.,2.,2.,2.,2.,2. --reuse_optimizer True```

- Rosenbrock submanifold 100dim
  - GAN
  
  ```python end_to_end.py --model GANModel --project_name RosenbrockModelDegenerate --work_space USERNAME --tags RosenbrockModelDegenerate,GANModel --optimizer TorchOptimizer --optimized_function RosenbrockModelDegenerate --step_data_gen 0.2 --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. --n_samples 20 --optimizer_config_file optimizer_config --lr 0.1```
  - FFJORD
  
  ```python end_to_end.py --model FFJORDModel --model_config_file ffjord_config --project_name RosenbrockModelDegenerate --work_space USERNAME --tags RosenbrockModelDegenerate,FFJORDModel --optimizer TorchOptimizer --optimized_function RosenbrockModelDegenerate --step_data_gen 0.2 --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. --n_samples 20 --optimizer_config_file optimizer_config --lr 0.1```
  - Numerical differentiation
  
  ```cd baseline_scripts && python baseline.py --project_name RosenbrockModelDegenerateBOCK --work_space USERNAME --optimized_function RosenbrockModelDegenerate --optimizer TorchOptimizer --tags num_diff,RosenbrockModelDegenerate --optimizer_config_file optimizer_config_num_diff  --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.```
  - Bayesin optimisation (BOCK)
  
  ```cd baseline_scripts && python baseline.py --project_name RosenbrockModelDegenerateTest --optimized_function RosenbrockModelDegenerate --work_space USERNAME --optimizer BOCKOptimizer --tags gp,RosenbrockModelDegenerate --optimizer_config_file optimizer_config_gp   --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.```
  - Guided evolutionary stratagies
  
  ```cd baseline_scripts && python baseline.py --project_name RosenbrockModelDegenerateLTSCMA --work_space USERNAME --optimized_function RosenbrockModelDegenerate --optimizer CMAGES --tags cma_es,rosenbrock_i --optimizer_config_file optimizer_config_cma_es --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. --p 100```
  - Void 
  
  ```cd void && python void_run.py --project_name RosenbrockModelDegenerateVoid --work_space USERNAME --optimizer_config_file optimizer_config_void --tags void,RosenbrockModelDegenerate --optimized_function RosenbrockModelDegenerate  --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.```
  - Learning to Simulate
  
  ```cd learn_to_sim && python lts_run.py --model LearnToSimModel --optimizer TorchOptimizer --optimized_function RosenbrockModelDegenerate --model_config_file lts_config --project_name RosenbrockModelDegenerateLTS --work_space USERNAME --tags lts,adam --epochs 2000 --n_samples 25 --n_samples_per_dim 3000 --init_psi 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. --reuse_optimizer True```

- Three Hump submanifold 40dim
  - GAN
  
  One can vary n_samples in \{3,5,10,20\} to reproduce various plots
  
  ```python end_to_end.py --model GANModel --project_name GaussianMixtureHumpModelDeepDegenerate --work_space USERNAME --tags GaussianMixtureHumpModelDeepDegenerate,GANModel --optimizer TorchOptimizer --optimized_function GaussianMixtureHumpModelDeepDegenerate --step_data_gen 0.2 --init_psi 2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. --n_samples 3 --optimizer_config_file optimizer_config --lr 0.1```
  - Numerical differentiation
  
  ```cd baseline_scripts && python baseline.py --project_name GaussianMixtureHumpModelDeepDegenerate --work_space USERNAME --optimized_function GaussianMixtureHumpModelDeepDegenerate --optimizer TorchOptimizer --tags num_diff,GaussianMixtureHumpModelDeepDegenerate --optimizer_config_file optimizer_config_num_diff  --init_psi 2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0. --h 0.2```
  - Bayesin optimisation (BOCK)
  
  ```cd baseline_scripts && python baseline.py --project_name GaussianMixtureHumpModelDeepDegenerate --work_space USERNAME --optimized_function GaussianMixtureHumpModelDeepDegenerate --optimizer BOCKOptimizer --tags gp,GaussianMixtureHumpModelDeepDegenerate --optimizer_config_file optimizer_config_gp  --init_psi 2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.```
 
- Neural network optimisation
  - GAN
  
  One can vary n_samples in \{20, 40, 60\} to reproduce various plots
  
  ```python end_to_end.py --model GANModel --project_name BostonNNTuning --work_space USERNAME --tags BostonNNTuning,GANModel --optimizer TorchOptimizer --optimized_function BostonNNTuning --step_data_gen 0.1 --init_psi  0.0215,0.0763,0.0879,0.0102,0.095,0.0508,0.088,0.101,0.0782,0.0684,0.0658,0.0509,0.0207,0.0618,0.0756,0.00784,0.0968,0.0685,0.0113,0.0745,0.00154,0.0772,0.0472,0.000906,0.0723,0.0779,0.0594,0.0785,0.0918,0.0634,0.0853,0.105,0.00407,0.0789,0.0035,0.0581,0.0375,0.0632,0.0669,0.00293,0.0901,0.0208,0.0388,0.0893,0.00104,0.0598,0.0745,0.08,0.0283,0.0106,0.0371,0.0667,0.0331,0.0356,0.0661,0.0554,0.084,0.0398,0.00286,0.0281,0.0246,0.0208,0.0358,0.033,0.0421,0.0505,0.00544,0.0269,0.00527,0.0569,0.00538,0.0786,0.102,0.0452,0.0444,0.105,0.0765,0.0689,0.0249,0.0933,0.037,0.0762,0.0882,0.0505,0.0688,0.0666,0.101,0.0857,0.0488,0.0303,22.5328  --n_samples 20 --optimizer_config_file optimizer_config --lr 0.1```
  - Numerical differentiation
  
  ```cd baseline_scripts && python baseline.py --project_name BostonNNTuning --work_space USERNAME --tags BostonNNTuning,num_diff --optimizer TorchOptimizer --optimized_function BostonNNTuning --init_psi  0.0215,0.0763,0.0879,0.0102,0.095,0.0508,0.088,0.101,0.0782,0.0684,0.0658,0.0509,0.0207,0.0618,0.0756,0.00784,0.0968,0.0685,0.0113,0.0745,0.00154,0.0772,0.0472,0.000906,0.0723,0.0779,0.0594,0.0785,0.0918,0.0634,0.0853,0.105,0.00407,0.0789,0.0035,0.0581,0.0375,0.0632,0.0669,0.00293,0.0901,0.0208,0.0388,0.0893,0.00104,0.0598,0.0745,0.08,0.0283,0.0106,0.0371,0.0667,0.0331,0.0356,0.0661,0.0554,0.084,0.0398,0.00286,0.0281,0.0246,0.0208,0.0358,0.033,0.0421,0.0505,0.00544,0.0269,0.00527,0.0569,0.00538,0.0786,0.102,0.0452,0.0444,0.105,0.0765,0.0689,0.0249,0.0933,0.037,0.0762,0.0882,0.0505,0.0688,0.0666,0.101,0.0857,0.0488,0.0303,22.5328 --h 0.05```
  - Bayesin optimisation (BOCK)
  
  ```cd baseline_scripts && python baseline.py --project_name BostonNNTuning --optimized_function BostonNNTuning --work_space USERNAME --optimizer BOCKOptimizer --tags gp,BostonNNTuning --optimizer_config_file optimizer_config_gp --init_psi  0.0215,0.0763,0.0879,0.0102,0.095,0.0508,0.088,0.101,0.0782,0.0684,0.0658,0.0509,0.0207,0.0618,0.0756,0.00784,0.0968,0.0685,0.0113,0.0745,0.00154,0.0772,0.0472,0.000906,0.0723,0.0779,0.0594,0.0785,0.0918,0.0634,0.0853,0.105,0.00407,0.0789,0.0035,0.0581,0.0375,0.0632,0.0669,0.00293,0.0901,0.0208,0.0388,0.0893,0.00104,0.0598,0.0745,0.08,0.0283,0.0106,0.0371,0.0667,0.0331,0.0356,0.0661,0.0554,0.084,0.0398,0.00286,0.0281,0.0246,0.0208,0.0358,0.033,0.0421,0.0505,0.00544,0.0269,0.00527,0.0569,0.00538,0.0786,0.102,0.0452,0.0444,0.105,0.0765,0.0689,0.0249,0.0933,0.037,0.0762,0.0882,0.0505,0.0688,0.0666,0.101,0.0857,0.0488,0.0303,22.5328```

