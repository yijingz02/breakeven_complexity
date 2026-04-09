import os
import numpy as np
from datagen import generate


Uin = 0.75
nu = 0.01
Re = int(2.0 * Uin / nu)
workdir = f'timeavg-Re{Re}'

for seed in range(5):
    os.makedirs(f'{workdir}/{seed}', exist_ok=True)
    for coarse_size in sorted(set(2.0 ** np.linspace(0.0, 4.0, 17)).union(2.0 ** np.linspace(0.0, 0.25, 5))):
        generate(f'{workdir}/{seed}/{coarse_size}',
                 config='inc-flow.ini',
                 seed=seed,
                 coarse_size=coarse_size,
                 iniswap=[
                          ('backend', 'precision', 'single'),
                          ('constants', 'nu', nu),
                          ('constants', 'Uin', Uin),
                          ('solver-time-integrator', 'tend', 1000.0),
                          ('solver-time-integrator', 'dt', min(0.07 / Uin, 0.2))],
                 resolution=64,
                 verbose=True,
                 progress_bar=True,
                 dump_gif=True)
