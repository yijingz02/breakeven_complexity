from glob import glob
from operator import itemgetter
import h5py
import numpy as np
from matplotlib import pyplot as plt


root = 'lowRe_'
sims = sorted(((folder, float(folder[len(root):].replace('p', '.')))
                for folder in glob(f'{root}*')), key=itemgetter(1))

stats = {'nRMSE': [], 'enstrophy': []}
overtime = {'nRMSE': [], 'enstrophy': []}
resolutions = []

for sim, h in sims:
    resolutions.append(h)
    with h5py.File(f'{sim}/flow.h5', 'r') as f:
        X = np.array(f['vorticity'])
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        x = (X*X).sum(1)
        X = np.stack([f['velocity_x'], f['velocity_y'], f['pressure']]).transpose((1, 0, 2, 3))
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    if sim == sims[0][0]:
        Z = X
        z = x
    stats['nRMSE'].append(np.linalg.norm(X - Z) / np.linalg.norm(Z))
    stats['enstrophy'].append(np.linalg.norm(x - z) / np.linalg.norm(z))
    overtime['nRMSE'].append(np.linalg.norm(X - Z, axis=1) / np.linalg.norm(Z, axis=1))
    overtime['enstrophy'].append(x)

os.makedirs('plots', exist_ok=True)
for label in ['nRMSE', 'enstrophy']:
    print(label)
    for h, stat, err in zip(resolutions, stats[label], overtime[label]):
        print('\t', h, '\t', stat)
        plt.plot(err, label=str(h))
    plt.title(label)
    plt.legend()
    plt.savefig(f'plots/{root}{label}.png', dpi=256)
    plt.clf()
