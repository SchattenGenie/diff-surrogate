model_config = {
    'task': "REVERSED_KL", #'WASSERSTEIN', # WASSERSTEIN, REVERSED_KL
    'y_dim': 1,
    'x_dim': 1,
    'psi_dim': 10,
    'noise_dim': 50,
    'lr': 1e-4,
    'batch_size': 64,
    'epochs': 3,
    'instance_noise_std': None
}
