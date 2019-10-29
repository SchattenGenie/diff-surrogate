model_config = {
    'task': "REVERSED_KL", #"CRAMER", #"CRAMER", #"REVERSED_KL",  # 'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 3,  # 2
    'x_dim': 1,  # 4
    'psi_dim': 10,  # obsolete
    'noise_dim': 50,
    'lr': 1e-4 * 8,
    'batch_size': 512,
    'epochs': 20,
    'iters_discriminator': 3,
    'iters_generator': 1,
    'instance_noise_std': None,
    'burn_in_period': None,
    'averaging_coeff': 0.,
    'dis_output_dim': 1,
    'grad_penalty': False,
    'attention_net_size': None,
    'gp_reg_coeff': None
}
