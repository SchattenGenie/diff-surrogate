**Note: This will not run on your own machine due to restrictions with kubernetes**


The code is structured as follows:
- file `model.py` contains all the code about how we communicate with SHiP. The most important class if FullSHiPModel, where we define the loss function.
- folder `local_train` contain all the code, which is neccesary to run optimistion. Optimisation can be run with different underlying algorithms:
  - L-GSO(ours)
  - Bayesian optimisation
  - Reinforcement learning
  - Numerial differences
  - Geneteic-like algorithms
  
For now lets concentrate on how to run the code for our algorithm.
### Step 1
  - Register on the https://www.comet.ml/ platform.
  - Go to https://www.comet.ml/[USERNAME]/settings/account amd generate API key.
  - Create file `~/.comet.config` and copy your api key in the file: `COMET_API_KEY=YOURKEY`

### Step 2
- create an environment with conda yml file: ```conda env create -f conda_env.yml```.
- activate environment with `conda activate lgso`.
- ```cd /diff-surrogate/local_train```. 

### Step 3
You can set various parameters of the algorithm
- Parameters of the optimiser are located in `local_train/optimizer_config.py`.
- Parameters of the optimising algorithm are located in `local_train/gan_config.py`.
- Parameters of the ship are set in the class `model.py` itself.
The rule of thumb is that as optimiser-> torch_model we usually select `Adam` or `SGD`. Another important parameter - learning rate we will set directly via command line. For now assume that `gan_config` has a good preselection of parameters.

### Step 4
Run optimisation:

`nohup python end_to_end.py --model GANModel --optimizer TorchOptimizer --optimized_function FullSHiPModel --project_name full_ship  --work_space YOURUSERNAME --tags TAGS --epochs 300 --n_samples 84 --n_samples_per_dim 485879 --init_psi "208.0,207.0,281.0,248.0,305.0,242.0,72.0,51.0,29.0,46.0,10.0,7.0,54.0,38.0,46.0,192.0,14.0,9.0,10.0,31.0,35.0,31.0,51.0,11.0,3.0,32.0,54.0,24.0,8.0,8.0,22.0,32.0,209.0,35.0,8.0,13.0,33.0,77.0,85.0,241.0,9.0,26.0" --reuse_optimizer True --step_data_gen 3. --lr 1. &`

Here we have: 
- **project_name**: this is the name of the project shown in your comet webpage.
- **work_space**: your comet.ml username
- **tags**: any tags(comma separated) you want to give to this run of the experiment.
- **epochs**: number of optimisation iterations.
- **n_samples**: number of parallel SHiP runs to estimate the gradient (do not go above 84).
- **n_samples_per_dim**: Number of events to run from input file for each SHiP run.
- **init_psi**: inital vector of parameters, in the format: length of magnet sections, [f_l, f_r, h_l, h_r,g_l,g_r] for each section of the magnet.
- **step_data_gen**: This is **very** important parameter, it tells the algorithm in which window to generate possible configuration on each iteration of the optimisation.
- **lr**: Learning rate. This parameter will affect **dramatically** on the how fast its possible to converge. Good values are in the range [0.1, 1].


The code for SHiP container and kubernetes service is located here:
https://github.com/SchattenGenie/ship_optimization_model
