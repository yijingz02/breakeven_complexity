import os
from glob import glob
from operator import itemgetter
import h5py
import numpy as np
from matplotlib import pyplot as plt
from flowgen import array2png


root = 'long_'
sims = sorted(((folder, float(folder[len(root):].replace('p', '.')))
    for folder in glob(f'{root}*')), key=itemgetter(1))[1:]

stats = {'velocities': [], 'avg-velo': [], 'enstrophy': [], 'vorticity': [], 'avg-vort': []}
overtime = {'velocities': [], 'avg-velo': [], 'enstrophy': [], 'vorticity': [], 'avg-vort': []}
resolutions = []

for sim, h in sims:
    resolutions.append(h)

    with h5py.File(f'{sim}/flow.h5', 'r') as f:
        data = {q: np.array(f[q]) for q in ['velocity_x', 'velocity_y', 'pressure', 'vorticity']}
        T, n, _ = data['pressure'].shape
        t = 0
        weight = 1.0 / np.arange(1, T-t+1)
        data['velocities'] = np.stack([data['velocity_x'], data['velocity_y']]).transpose(1, 0, 2, 3)
        for q in ['velocities', 'vorticity']:
            data[f'avg-{q[:4]}'] = data[q][t:].cumsum(0) * (weight[:,None,None,None] if len(data[q].shape) == 4 else weight[:,None,None])
        data['enstrophy'] = np.square(data['vorticity']).reshape(T, n*n).sum(1)

    if sim == sims[0][0]:
        target = {q: v for q, v in data.items()}
        for i, q in enumerate(['velocity_x', 'velocity_y']):
            array2png(target['velocities'][t:T,i].mean(0), f'plots/{root}avg-{q}.png')
        array2png(target['avg-vort'][-1], f'plots/{root}avg-vorticity.png')
        for q in ['velocities', 'avg-velo', 'vorticity', 'avg-vort']:
            target[q] = target[q].reshape(T-t if 'avg' in q else T, 2*n*n if 'velo' in q else n*n)
    for q in ['velocities', 'avg-velo', 'vorticity', 'avg-vort']:
        data[q] = data[q].reshape(T-t if 'avg' in q else T, 2*n*n if 'velo' in q else n*n)

    for q in ['velocities', 'vorticity']:
        stats[q].append(np.linalg.norm(data[q] - target[q]) / np.linalg.norm(target[q]))
        avgq = f'avg-{q[:4]}'
        stats[avgq].append(np.linalg.norm(data[avgq][-1] - target[avgq][-1]) / np.linalg.norm(target[avgq][-1]))
    stats['enstrophy'].append(np.linalg.norm(data['enstrophy'] - target['enstrophy']) / np.linalg.norm(target['enstrophy']))

    for q in ['velocities', 'vorticity', 'avg-velo', 'avg-vort']:
        overtime[q].append(np.linalg.norm(data[q] - target[q], axis=1) / np.linalg.norm(target[q], axis=1))
    overtime['enstrophy'].append(data['enstrophy'])

os.makedirs('plots', exist_ok=True)
for label in overtime.keys():
    print(label)
    for h, stat, err in zip(resolutions, stats[label], overtime[label]):
        print('\t', h, '\t', stat)
        plt.plot(err, label=str(h))
    plt.title(label)
    plt.legend()
    plt.savefig(f'plots/{root}{label}.png', dpi=256)
    plt.clf()
