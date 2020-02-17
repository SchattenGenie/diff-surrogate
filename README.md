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

## To reproduce experiments
- Three Hump Model
```code```
- Rosenbrock 10dim
```code```
- Rosenbrock submanifold 100dim
```code```
- Three Hump submanifold 40dim
```code```
- Neural network optimisation
```code```


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
