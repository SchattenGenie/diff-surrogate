model_config = {
    'task': "CRAMER", #"REVERSED_KL", #"CRAMER", #"CRAMER", #"REVERSED_KL",  # 'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 1,  # 2
    'x_dim': 1,  # 4
    'psi_dim': 20,  # obsolete
    'noise_dim': 100,
    'lr': 1e-4 * 8,
    'batch_size': 512,
    'epochs': 30,
    'iters_discriminator': 1,
    'iters_generator': 1,
    'instance_noise_std': None,
    'burn_in_period': None,
    'averaging_coeff': 0.,
    'dis_output_dim': 256,
    'grad_penalty': True,
    'attention_net_size': None,
    'gp_reg_coeff': 10
}
