model_config = {
    'task': "REVERSED_KL",  # 'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 3,
    'x_dim': 1,
    'psi_dim': 12,  # obsolete
    'noise_dim': 50,
    'lr': 1e-4 * 8,
    'batch_size': 512,
    'epochs': 15,
    'iters_discriminator': 3,
    'iters_generator': 1,
    'instance_noise_std': None
}
