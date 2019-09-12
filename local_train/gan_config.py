model_config = {
    'task': "REVERSED_KL", #"CRAMER", #"REVERSED_KL",  # 'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 1,
    'x_dim': 1,
    'psi_dim': 10,  # obsolete
    'noise_dim': 100,
    'lr': 1e-4 * 4,
    'batch_size': 256,
    'epochs': 30,
    'iters_discriminator': 5,
    'iters_generator': 1,
    'instance_noise_std': None,
    'burn_in_period': None,
    'averaging_coeff': 0.,
    'dis_output_dim': 1,
    'grad_penalty': False,
    'attention_net_size': 24
}
