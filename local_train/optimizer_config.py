optimizer_config = {
    'lr': 0.1,
    'num_repetitions': 2000,
    'max_iters': 10,
#     'x_step': 1.,
     'line_search_options': {
         "method": 'Wolfe',
         'c0': 1.,
         'c1': 1e-4,
         'c2': 0.5,
     },
    'torch_model': 'RMSprop',
}
