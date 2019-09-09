model_config = {
    'task': 'WASSERSTEIN', #"REVERSED_KL",  # 'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 1,
    'x_dim': 1,
    'psi_dim': 12,  # obsolete
    'noise_dim': 100,
    'lr': 1e-4 * 4,
    'batch_size': 256,
    'epochs': 1,
    'iters_discriminator': 5,
    'iters_generator': 1,

    'instance_noise_std': None
}
