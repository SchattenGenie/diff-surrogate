optimizer_config = {
    'lr': 1e-1,
    'num_repetitions': 5000,
    'max_iters': 3,
    'x_step': 1.,
    'line_search_options': {
        "method": 'Wolfe',
        'c1': 1e-4,
        'c2': 0.5,
    },
    'torch_model': 'RMSprop',
}
