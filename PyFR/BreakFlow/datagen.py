import math
import os
import subprocess
from glob import glob
from operator import itemgetter
from time import perf_counter
import h5py
from pyfr.inifile import Inifile
from meshgen import random_rectangle_mesh, square_test_mesh


def actual_dt_out_matches_expected(tstart, tend, dt, dt_out, dt_min=1e-12, rounding=2):

    expected = []
    tcurr = tstart
    while tcurr <= tend:
        expected.append(tcurr)
        tcurr += dt_out

    actual = [tstart]
    tcurr = tstart
    tout_last = tstart
    tol = 5 * dt_min
    while tcurr <= tend:
        tcurr += dt
        if tcurr - tout_last >= dt_out - tol:
            actual.append(tcurr)
            tout_last = tcurr

    return expected == [round(t, rounding) for t in actual]


def generate(folder, 
             config='inc-flow.ini',     # base .ini file
             test_mesh=False,           # use a single square obstacle 
             coarse_size=1.0,           # also scales fine_size & obstacle_size
             dt_step=0.01,              # amount to reduce base dt by if NaNs
             dt_min=0.01,               # smallest base dt to try
             iniswap=[],                # parameters to change in .ini file
             resolution=64,             # structured data resolution
             verbose=False,
             progress_bar=False,
             dump_gif=False, 
             **kwargs):

    mshfile = f'{folder}/mesh.msh'
    pyfrmfile = f'{folder}/mesh.pyfrm'
    inifile = f'{folder}/cfg.ini'
    resfile = f'{folder}/residual.csv'
    os.makedirs(folder, exist_ok=True)

    cfg = Inifile.load(config)
    cfg.set('soln-plugin-writer', 'basedir', folder)
    cfg.set('soln-plugin-pseudostats', 'file', resfile)
    for section, option, value in iniswap:
        cfg.set(section, option, value)
    tstart = cfg.getfloat('solver-time-integrator', 'tstart')
    tend = cfg.getfloat('solver-time-integrator', 'tend')
    dt_base = cfg.getfloat('solver-time-integrator', 'dt')
    dt_out = cfg.getfloat('soln-plugin-writer', 'dt-out')

    wallclock = -perf_counter()
    meshgen = square_test_mesh if test_mesh else random_rectangle_mesh
    meshgen(output=mshfile, coarse_size=coarse_size, verbosity=2 if verbose else 0, **kwargs)   # pass seed via kwargs
    subprocess.run(['pyfr', 'import', mshfile, pyfrmfile], check=True)                          # converts to PyFR mesh

    pyfrcommand = ['pyfr']
    if progress_bar:
        pyfrcommand.append('-p')
    pyfrcommand.extend(['run', '-b', 'cuda'])
    meshsolncfg = [pyfrmfile, inifile]
    while dt_base >= dt_min:

        try:
            denom = int(dt_out / (dt_base * coarse_size))
            while True: # figures out largest viable dt
                dt = dt_out / denom
                if actual_dt_out_matches_expected(tstart, tend, dt, dt_out):
                    break
                denom += 1
            cfg.set('solver-time-integrator', 'dt', dt)
            cfg.set('solver-time-integrator', 'pseudo-dt', dt / 10.0)
            with open(inifile, 'w') as f:
                f.write(cfg.tostr())
            subprocess.run(pyfrcommand + meshsolncfg, check=True)
            break
        
        # figures out last timestep without NaNs and restarts
        except subprocess.CalledProcessError:               
            dt_base -= dt_step
            if dt_base < dt_min:
                raise
            pyfrcommand[1+progress_bar] = 'restart'
            pyfrsfiles = sorted(((pyfrsfile, float(pyfrsfile[:-6].rsplit('.', 1)[0].rsplit('-', 1)[1]))
                                 for pyfrsfile in glob(f'{folder}/*.pyfrs')),
                                 key=itemgetter(1))
            last = 0.0
            if os.path.isfile(resfile):
                with open(resfile, 'r') as f:
                    f.readline()
                    for line in f:
                        split = line.split(',')
                        if not any(e == 'nan' for e in split[-3:]):
                            last = float(split[1])
            for pyfrsfile, t in reversed(pyfrsfiles):
                if t <= last + 0.1 * dt:
                    break
            meshsolncfg = [pyfrmfile, pyfrsfile, inifile]

    # convers PyFR save files to VTU format for ParaView processing
    wallclock += perf_counter()
    for pyfrsfile in glob(f'{folder}/*.pyfrs'):
        subprocess.run(['pyfr', 'export', pyfrmfile, pyfrsfile, f'{pyfrsfile[:-6]}.vtu'])

    # runs Python code in a subprocess to avoid non-deterministic ParaView errors...
    flowgen = ['python', 'flowgen.py', folder, '--resolution', str(resolution)]
    if dump_gif:
        flowgen.append('--dump-gif')
    if verbose:
        flowgen.append('--verbose')
    subprocess.run(flowgen)
    for vtufile in glob(f'{folder}/*.vtu'):
        os.remove(vtufile)

    with h5py.File(f'{folder}/flow.h5', 'r+') as f:
        f.create_dataset('wallclock', data=wallclock)


if __name__ == '__main__':

    coarse_size = 1.0
    seed = 0
    generate(f'{seed}', 
             config='inc-flow.ini',
             seed=seed,
             coarse_size=coarse_size, 
             iniswap=[('backend', 'precision', 'single'),
                      ('constants', 'nu', 0.01), 
                      ('solver-time-integrator', 'dt', 0.07)],
             resolution=64, 
             verbose=True,
             progress_bar=True,
             dump_gif=True)
