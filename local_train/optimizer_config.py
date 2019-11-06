optimizer_config = {
    'lr': 1,
    'num_repetitions': 10000,
    'max_iters': 1,
     # 'line_search_options': {
     #     "method": 'Wolfe',
     #     'c0': 1.,
     #     'c1': 1e-4,
     #     'c2': 0.5,
     # },
     'optim_params': {"momentum": 0.9},
    'torch_model': 'SGD',
}
