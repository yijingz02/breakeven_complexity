import os
import subprocess
from glob import glob
from operator import itemgetter
from meshgen import random_rectangle_mesh, square_test_mesh


def generate(folder, 
             config='inc-flow.ini',     # base .ini file
             test_mesh=False,           # use a single square obstacle 
             coarse_size=1.0,           # also scales fine_size & obstacle_size
             dts=(0.05, 0.04, 0.03),    # base timesteps to try
             iniswap={},                # parameters to change in .ini file
             dt_out=5.0, 
             resolution=64, 
             dump_gif=False, 
             **kwargs):

    mshfile = f'{folder}/mesh.msh'
    pyfrmfile = f'{folder}/mesh.pyfrm'
    inifile = f'{folder}/cfg.ini'
    csvfile = f'{folder}/residual.csv'
    os.makedirs(folder, exist_ok=True)

    meshgen = square_test_mesh if test_mesh else random_rectangle_mesh
    meshgen(output=mshfile, coarse_size=coarse_size, **kwargs)          # pass seed via kwargs
    subprocess.run(['pyfr', 'import', mshfile, pyfrmfile], check=True)  # converts to PyFR mesh

    pyfrcommand = 'run'
    meshsolncfg = [pyfrmfile, inifile]
    for dt_base in dts: 

        try:
            dt = dt_out / int(dt_out / (dt_base*coarse_size))
            cfgswap = {'dt': dt,
                       'pseudo-dt': dt/10.0,
                       'dt-out': dt_out,
                       'file': csvfile,
                       'basedir': folder}
            cfgswap.update(iniswap)
            with open(config, 'r') as f:
                with open(inifile, 'w') as g:
                    for line in f:
                        key = line.split(' = ')[0]
                        if key in cfgswap:
                            g.write(f'{key} = {cfgswap[key]}\n')
                        else:
                            g.write(line)
            subprocess.run(['pyfr', '-p', pyfrcommand, '-b', 'cuda'] + meshsolncfg, check=True)
            break

        except subprocess.CalledProcessError:   # figures out last timestep without NaNs and restarts
            if dt_base == dts[-1]:
                raise
            pyfrcommand = 'restart'
            pyfrsfiles = sorted(((pyfrsfile, float(pyfrsfile[:-6].rsplit('.', 1)[0].rsplit('-', 1)[1]))
                                 for pyfrsfile in glob(f'{folder}/*.pyfrs')),
                                 key=itemgetter(1))
            with open(config, 'r') as f:
                for line in f:
                    if line[:9] == 'nsteps = ':
                        nsteps = int(line[9:])
                        break
            last = (int(pyfrsfiles[-1][1] / (dt * nsteps))+1) * dt * nsteps
            for pyfrsfile, t in reversed(pyfrsfiles):
                if t <= last - dt * nsteps:
                    break
            meshsolncfg = [pyfrmfile, pyfrsfile, inifile]

    # convers PyFR save files to VTU format for ParaView processing
    for pyfrsfile in glob(f'{folder}/*.pyfrs'):
        subprocess.run(['pyfr', 'export', pyfrmfile, pyfrsfile, f'{pyfrsfile[:-6]}.vtu'])

    # runs Python code in a subprocess to avoid non-deterministic ParaView errors...
    subprocess.run(['python', 'flowgen.py', folder, '--resolution', str(resolution)] + ['--dump-gif'] if dump_gif else [])
    for vtufile in glob(f'{folder}/*.vtu'):
        os.remove(vtufile)


if __name__ == '__main__':

    for coarse_size in [2.0, 1.7, 1.4, 1.2, 1.0]:
        generate(f'long_{str(coarse_size).replace(".", "p")}', 
                 config='inc-flow.ini',
                 seed=0,
                 coarse_size=coarse_size, 
                 iniswap={'tend': 1000.0},
                 resolution=256, 
                 dump_gif=True)

    for coarse_size in [2.0, 1.7, 1.4, 1.2, 1.0, 0.85, 0.7, 0.5, 0.35, 0.25]:
        generate(f'lowRe_{str(coarse_size).replace(".", "p")}', 
                 config='inc-flow.ini',
                 seed=0,
                 coarse_size=coarse_size, 
                 dts=(0.07, 0.06, 0.05, 0.04),
                 iniswap={'nu': 0.01},
                 resolution=256, 
                 dump_gif=True)
