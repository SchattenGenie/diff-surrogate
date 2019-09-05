model_config = {
    'task': "REVERSED_KL",  # 'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 1,
    'x_dim': 2,
    'psi_dim': 12,  # obsolete
    'noise_dim': 50,
    'lr': 1e-4 * 8,
    'batch_size': 512,
    'epochs': 40,
    'iters_discriminator': 5,
    'iters_generator': 1,
    'instance_noise_std': None
}
