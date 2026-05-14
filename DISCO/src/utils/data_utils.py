from typing import Tuple, List, TypeVar, Iterator, Dict, Sized, Optional
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler

from src.utils.database import subsample
from src.the_well.datasets import GenericWellDataset


PDEBENCH_PATH = "..."  # TODO: replace this with your own path to the PDEBench dataset
THE_WELL_PATH = "..."  # TODO: replace this with your own path to the the_well dataset
CUSTOM_NPY_PATH = "data"

BURGERS_SPECS = {
    'main_path': PDEBENCH_PATH + '/1D/Burgers/Train',
    'include_string': '',
    'resolution': (1024,),
    'in_channels': 1,
    'spatial_ndims': 1,
    'boundary_conditions': 'periodic',
    'n_steps': 200,
    'group': 'PDEBench',
}
SWE_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/shallow-water',
    'include_string': '',
    'resolution': (128,128),
    'in_channels': 1,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    'n_steps': 100,
    'group': 'PDEBench',
}
DIFFRE2D_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/diffusion-reaction',
    'include_string': '',
    'resolution': (128,128),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'neumann',
    'n_steps': 100,
    'group': 'PDEBench',
}
INCOMPNS_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/NS_incom',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 3,
    'spatial_ndims': 2,
    'boundary_conditions': 'dirichlet',
    'n_steps': 1000,
    'group': 'PDEBench',
}
COMPNS128_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': '128',
    'resolution': (128,128),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
COMPNS512_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': '512',
    'resolution': (512,512),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
COMPNS_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Turb',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
EULER_PBC_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/euler_multi_quadrants_periodicBC',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 5,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 100,
    'group': 'the_well',
}
EULER_OBC_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/euler_multi_quadrants_openBC',
    'include_string': '',
    'resolution': (512,512),
    'in_channels': 5,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    'n_steps': 100,
    'group': 'the_well',
}
ACTIVE_MATTER_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/active_matter',
    'include_string': '',
    'resolution': (256,256),
    'in_channels': 11,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 81,
    'group': 'the_well',
}
CONVECTIVE_ENVELOPPE_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/convective_envelope_rsg',
    'include_string': '',
    # 'resolution': (256,128,256),  # stored resolution
    'resolution': (64,32,64),
    'in_channels': 6,
    'spatial_ndims': 3,
    'boundary_conditions': 'custom',  # seems Neumann, but should be confirmed
    'n_steps': 100,
    'group': 'the_well',
}
GRAY_SCOTT_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/gray_scott_reaction_diffusion',
    'include_string': '',
    'resolution': (128,128),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 1001,
    'group': 'the_well',
}
HELMHOLTZ_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/helmholtz_staircase',
    'include_string': '',
    'resolution': (1024,256),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'neumann',
    'n_steps': 50,
    'group': 'the_well',
}
MHD_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/MHD_64',
    'include_string': '',
    'resolution': (64,64,64),
    'in_channels': 7,
    'spatial_ndims': 3,
    'boundary_conditions': 'periodic',
    'n_steps': 100,
    'group': 'the_well',
}
RAYLEIGH_BENARD_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/rayleigh_benard_uniform',
    'include_string': '',
    'resolution': (512,128),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic x dirichlet',
    'n_steps': 200,
    'group': 'the_well',
} 
RAYLEIGH_TAYLOR_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/rayleigh_taylor_instability',
    'include_string': '',
    # 'resolution': (128,128,128),  # stored resolution 
    'resolution': (64,64,64),
    'in_channels': 4,
    'spatial_ndims': 3,
    'boundary_conditions': 'periodic x dirichlet',
    'n_steps': 120,  # the github says 119
    'group': 'the_well',
}
SHEARFLOW_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/shear_flow',
    'include_string': '',
    'resolution': (256,512),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 200,
    'group': 'the_well',
}
SUPERNOVA_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/supernova_explosion_64',
    'include_string': '',
    'resolution': (64,64,64),
    'in_channels': 6,
    'spatial_ndims': 3,
    'boundary_conditions': 'open',
    'n_steps': 59,
    'group': 'the_well',
}
GRAVITY_COOLING_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/turbulence_gravity_cooling',
    'include_string': '',
    'resolution': (64,64,64),
    'in_channels': 6,
    'spatial_ndims': 3,
    'boundary_conditions': 'open',
    'n_steps': 50,
    'group': 'the_well',
}
RADIATIVE_LAYER_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/turbulent_radiative_layer_2D',
    'include_string': '',
    'resolution': (128,384),
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic x neumann',
    'n_steps': 101,
    'group': 'the_well',
}
COMPNS_M1_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': 'M1.0',
    'resolution': (512,512),  # the maximum resolution achievable
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
COMPNS_M01_SPECS = {
    'main_path': PDEBENCH_PATH + '/2D/CFD/2D_Train_Rand',
    'include_string': 'M0.1',
    'resolution': (512,512),  # the maximum resolution achievable
    'in_channels': 4,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 21,
    'group': 'PDEBench',
}
EULER_OBC_FINETUNE_SPECS = {
    'main_path': THE_WELL_PATH + '/datasets/euler_multi_quadrants_openBC',
    'include_string': 'gamma_1.3',
    'resolution': (512,512),
    'in_channels': 5,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    'n_steps': 100,
    'group': 'the_well',
}
CUSTOM_NPY_SPECS = {
    'main_path': CUSTOM_NPY_PATH,
    'include_string': '',
    'resolution': (128, 128),  # Change this to match your data resolution
    'in_channels': 1,  # Change this to match your number of fields/channels
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 20,  # This will be overridden by training config
    'group': 'custom_npy',
}

# Breakflow dataset (obstacle flow with mask). NPY files expected under data/breakflow
BREAKFLOW_SPECS = {
    'main_path': CUSTOM_NPY_PATH + '/breakflow',
    'include_string': '',
    'resolution': (64, 64),
    # Breakflow trajectories in this workspace are single-channel fields with
    # shape [N, T, 64, 64] (train) and a longer held-out test rollout.
    'in_channels': 1,
    'spatial_ndims': 2,
    'boundary_conditions': 'open',
    # Training/validation trajectories are 40 steps long, while the held-out
    # breakflow test trajectories are 200 steps long.
    'train_n_steps': 40,
    'test_n_steps': 200,
    'n_steps': 200,
    'group': 'custom_npy',
}

# Navier-Stokes dataset
NAVIERSTOKES_SPECS = {
    'main_path': CUSTOM_NPY_PATH + '/navier_stokes',
    'include_string': '',
    'resolution': (64, 64),
    'in_channels': 1,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 10,
    'group': 'custom_npy',
}

# Kuramoto-Sivashinsky dataset
KURAMOTOSIVASHINSKY_SPECS = {
    'main_path': CUSTOM_NPY_PATH + '/kuramoto_sivashinsky',
    'include_string': '',
    'resolution': (64, 64),
    'in_channels': 1,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    'n_steps': 50,
    'group': 'custom_npy',
}

# Gray Scott dataset (two fields: u and v)
GRAYSCOTT_SPECS = {
    'main_path': CUSTOM_NPY_PATH + '/gray_scott',
    'include_string': '',
    'resolution': (64, 64),
    'in_channels': 2,
    'spatial_ndims': 2,
    'boundary_conditions': 'periodic',
    # 'n_steps': 50,
    'n_steps': 200,  # Increased from 50 to reduce sliding window count. With 200-timestep data: ~50 windows/sample instead of 150
    'group': 'custom_npy',
}
DATASET_SPECS = {
    'burgers': BURGERS_SPECS,
    'swe': SWE_SPECS,
    'diffre2d': DIFFRE2D_SPECS,
    'incompNS': INCOMPNS_SPECS,
    'compNS128': COMPNS128_SPECS,
    'compNS512': COMPNS512_SPECS,
    'compNS': COMPNS_SPECS,
    'active_matter': ACTIVE_MATTER_SPECS,
    'convective_envelope': CONVECTIVE_ENVELOPPE_SPECS,
    'euler_obc': EULER_OBC_SPECS,
    'euler_pbc': EULER_PBC_SPECS,
    'gray_scott': GRAY_SCOTT_SPECS,
    'helmholtz': HELMHOLTZ_SPECS,
    'mhd': MHD_SPECS,
    'rayleigh_benard': RAYLEIGH_BENARD_SPECS,
    'rayleigh_taylor': RAYLEIGH_TAYLOR_SPECS,
    'shear_flow': SHEARFLOW_SPECS,
    'supernova': SUPERNOVA_SPECS,
    'gravity_cooling': GRAVITY_COOLING_SPECS,
    'radiative_layer': RADIATIVE_LAYER_SPECS,
    # finetuning datasets
    'compNSM1.0': COMPNS_M1_SPECS,
    'compNSM0.1': COMPNS_M01_SPECS,
    'euler_obc_finetune': EULER_OBC_FINETUNE_SPECS,
    # custom NPY datasets
    'custom_npy': CUSTOM_NPY_SPECS,
    'navierstokes': NAVIERSTOKES_SPECS,
    'kuramotosivashinsky': KURAMOTOSIVASHINSKY_SPECS,
    'grayscott': GRAYSCOTT_SPECS,
    'breakflow': BREAKFLOW_SPECS,
}
 

broken_paths = [
    PDEBENCH_PATH + '/1D/Burgers/Train/1D_Burgers_Sols_Nu4.0.hdf5'
]


class BaseHDF5DirectoryDataset(Dataset):
    """
    Base class for data loaders. Returns data in T x B x C x H x W format.

    Note - doesn't currently normalize because the data is on wildly different
    scales but probably should.

    Split is provided so I can be lazy and not separate out HDF5 files.

    Takes in path to directory of HDF5 files to construct dset. 

    Args:
        path (str): Path to directory of HDF5 files
        include_string (str): Only include files with this string in name
        n_steps (int): Number of steps to include in each sample
        dt (int): Time step between samples
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        subname (str): Name to use for dataset
        split_level (str): 'sample' or 'file' - whether to split by samples within a file
                        (useful for data segmented by parameters) or file (mostly INS right now)
    """
    def __init__(self, path, resolution, include_string='', n_steps=1, dt=1, split='train', 
                 train_val_test=None, subname=None, extra_specific=False):
        super().__init__()
        self.path = path
        self.resolution = resolution
        self.split = split
        self.extra_specific = extra_specific # Whether to use parameters in name
        if subname is None:
            self.subname = path.split('/')[-1]
        else:
            self.subname = subname
        self.dt = 1
        self.n_steps = n_steps
        self.include_string = include_string
        # self.time_index, self.sample_index = self._set_specifics()
        self.train_val_test = train_val_test
        self.partition = {'train': 0, 'valid': 1, 'test': 2}[split]
        self.time_index, self.sample_index, self.field_names, self.type, self.split_level = self._specifics()
        self._get_directory_stats(path)
        if self.extra_specific:
            self.title = self.more_specific_title(self.type, path, include_string)
        else:
            self.title = self.type
        self.dataset_name = self.get_name() + self.include_string

    def get_name(self, full_name=False):
        if full_name:
            return self.subname + '_' + self.type
        else:
            return self.type
    
    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        return type

    @staticmethod
    def _specifics():
        # Sets self.field_names, self.dataset_type
        raise NotImplementedError # Per dset
    
    def get_per_file_dsets(self):
        if self.split_level == 'file' or len(self.files_paths) == 1:
            return [self]
        else:
            sub_dsets = []
            for file in self.files_paths:
                subd = self.__class__(self.path, file, n_steps=self.n_steps, dt=self.dt, split=self.split,
                               train_val_test=self.train_val_test, subname=self.subname,
                                 extra_specific=True)
                sub_dsets.append(subd)
            return sub_dsets

    def _get_specific_stats(self, f):
        raise NotImplementedError # Per dset
    
    def _get_specific_bcs(self, f):
        raise NotImplementedError # Per dset

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        raise NotImplementedError # Per dset - should be (x=(-history:local_idx+dt) so that get_item can split into x, y

    def _get_directory_stats(self, path):
        self.files_paths = glob.glob(path + "/*.h5") + glob.glob(path + "/*.hdf5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        self.file_steps = []
        self.file_nsteps = []
        self.file_samples = []
        self.split_offsets = []
        self.offsets = [0]
        file_paths = []
        for file in self.files_paths:
            # Total hack to avoid complications from folder with two sizes. 
            if len(self.include_string) > 0 and self.include_string not in file:
                continue
            elif file in broken_paths:
                continue
            else:
                file_paths.append(file)
                try:
                    with h5py.File(file, 'r') as _f:
                        samples, steps = self._get_specific_stats(_f)
                        if steps-self.n_steps-(self.dt-1) < 1:
                            print('WARNING: File {} has {} steps, but n_steps is {}. Setting file steps = max allowable.'.format(file, steps, self.n_steps))
                            file_nsteps = steps - self.dt
                        else:
                            file_nsteps = self.n_steps
                        self.file_nsteps.append(file_nsteps)
                        self.file_steps.append(steps-file_nsteps-(self.dt-1))
                        if self.split_level == 'sample':
                            # Compute which are in the given partition
                            partition = self.partition
                            sample_per_part = np.ceil(np.array(self.train_val_test)*samples).astype(int)
                            # Make sure rounding works
                            sample_per_part[2] = max(samples - sample_per_part[0] - sample_per_part[1], 0)
                            # I forget where the file steps formula came from, but offset by steps per sample
                            # * samples of previous partitions
                            self.split_offsets.append(self.file_steps[-1]*sum(sample_per_part[:partition]))
                            split_samples = sample_per_part[partition]
                        else:
                            split_samples = samples
                        self.file_samples.append(split_samples)
                        self.offsets.append(self.offsets[-1]+(steps-file_nsteps-(self.dt-1))*split_samples)
                except:
                    print('WARNING: Failed to open file {}. Continuing without it.'.format(file))
                    raise RuntimeError('Failed to open file {}'.format(file))
        # print(self.file_steps, self.file_samples)
        self.files_paths = file_paths
        self.offsets[0] = -1 # Just to make sure it doesn't put us in file -1
        self.files = [None for _ in self.files_paths]
        self.len = self.offsets[-1]
        if self.split_level == 'file':
        # Figure out our split offset - by sample
            if self.train_val_test is None:
                print('WARNING: No train/val/test split specified. Using all data for training.')
                self.split_offset = 0
                self.len = self.offsets[-1]
            else:
                # print('Using train/val/test split: {}'.format(self.train_val_test))
                total_samples = sum(self.file_samples)
                ideal_split_offsets = [int(self.train_val_test[i]*total_samples) for i in range(3)]
                # Doing this the naive way because I only need to do it once
                # Iterate through files until we get enough samples for set
                end_ind = 0
                for i in range(self.partition+1):
                    run_sum = 0
                    start_ind = end_ind
                    for samples, steps in zip(self.file_samples, self.file_steps):
                        run_sum += samples
                        if run_sum <= ideal_split_offsets[i]:
                            end_ind += samples * (steps)
                            if run_sum == ideal_split_offsets[i]:
                                break
                        else:
                            end_ind += np.abs((run_sum - samples) - ideal_split_offsets[i]) * (steps)
                            break
                self.split_offset = start_ind
                self.len = end_ind - start_ind
            # else:
                
                
    def _open_file(self, file_ind):
        _file = h5py.File(self.files_paths[file_ind], 'r')
        self.files[file_ind] = _file

    def __getitem__(self, index):
        if self.split_level == 'file':
            index = index + self.split_offset

        file_idx = int(np.searchsorted(self.offsets, index, side='right')-1) #which file we are on
        # print('sample from:', self.files_paths[file_idx])
        nsteps = self.file_nsteps[file_idx] # Number of steps per sample in given file
        local_idx = index - max(self.offsets[file_idx], 0) # First offset is -1
        if self.split_level == 'sample':
            sample_idx = (local_idx + self.split_offsets[file_idx]) // self.file_steps[file_idx]
        else:
            sample_idx = local_idx // self.file_steps[file_idx]
        time_idx = local_idx % self.file_steps[file_idx]

        #open image file
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        #if we are on the last image in a file shift backward. Double counting until I bother fixing this.
        time_idx = time_idx - self.dt if time_idx >= self.file_steps[file_idx] else time_idx
        time_idx += nsteps
        try:
            # print(self.files[file_idx], sample_idx, time_idx, index)
            trajectory = self._reconstruct_sample(self.files[file_idx], sample_idx, time_idx, nsteps)
            bcs = self._get_specific_bcs(self.files[file_idx])
        except:
            raise RuntimeError(f'Failed to reconstruct sample for file {self.files_paths[file_idx]} sample {sample_idx} time {time_idx}')
        return {
            'input_fields': subsample(trajectory[:-1,...], self.resolution),
            'output_fields': subsample(trajectory[[-1],...], self.resolution),  # single step in the future
            # 'field_labels': torch.tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]),
            'boundary_conditions': torch.as_tensor(bcs),
            'name': self.dataset_name,
            'file': self.files_paths[file_idx].split('/')[-1],
            'index': torch.as_tensor(index),
            # 'file_index': file_idx,
            # 'name': dataset.get_name() + dataset.include_string  # not really supposed to be unique
        }
        # return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1]

    def __len__(self):
        return self.len


class SWEDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ['h']
        type = 'swe'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]['data'].shape[0]
        return len(samples), steps
    
    def _get_specific_bcs(self, f):
        return [0, 0] # Non-periodic

    def _reconstruct_file(self, file, sample_idx):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][:].transpose(0, 3, 1, 2)

    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][time_idx-n_steps*self.dt:time_idx+self.dt].transpose(0, 3, 1, 2)


class DiffRe2DDataset(BaseHDF5DirectoryDataset):
    @staticmethod
    def _specifics():
        time_index = 0
        sample_index = None
        field_names = ['activator', 'inhibitor']
        type = 'diffre2d'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = list(f.keys())
        steps = f[samples[0]]['data'].shape[0]
        return len(samples), steps
    
    def _get_specific_bcs(self, f):
        return [0, 0] # Non-periodic

    def _reconstruct_file(self, file, sample_idx):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][:].transpose(0, 3, 1, 2)
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        samples = list(file.keys())
        return file[samples[sample_idx]]['data'][time_idx-n_steps*self.dt:time_idx+self.dt].transpose(0, 3, 1, 2)


class IncompNSDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, "particles"
    """
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx', 'Vy', 'particles']
        type = 'incompNS'
        split_level = 'file'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f['velocity'].shape[0] 
        steps = f['velocity'].shape[1]# Per dset
        return samples, steps

    def _reconstruct_file(self, file, sample_idx):
        velocity = file['velocity'][sample_idx, :]
        particles = file['particles'][sample_idx, :]
        comb =  np.concatenate([velocity, particles], -1)
        return comb.transpose((0, 3, 1, 2))
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        velocity = file['velocity'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        particles = file['particles'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        comb =  np.concatenate([velocity, particles], -1)
        return comb.transpose((0, 3, 1, 2))
    
    def _get_specific_bcs(self, f):
        return [0, 0] # Non-periodic


class CompNSDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx', 'Vy', 'density', 'pressure']
        type = 'compNS'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f['Vx'].shape[0] 
        steps = f['Vx'].shape[1]# Per dset
        return samples, steps

    def more_specific_title(self, type, path, include_string):
        """
        Override this to add more info to the dataset name
        """
        cns_path = self.include_string.split('/')[-1].split('_')
        ic = cns_path[2]
        m = cns_path[3]
        res = cns_path[-2]

        return f'{type}_{ic}_{m}_res{res}'

    def _reconstruct_file(self, file, sample_idx):
        vx = file['Vx'][sample_idx, :]
        vy = file['Vy'][sample_idx, :]
        density = file['density'][sample_idx, :]
        p = file['pressure'][sample_idx, :]

        comb =  np.stack([vx, vy, density, p], 1)
        return comb#.transpose((0, 3, 1, 2))
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file['Vx'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        vy = file['Vy'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        density = file['density'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        p = file['pressure'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]

        comb =  np.stack([vx, vy, density, p], 1)
        return comb#.transpose((0, 3, 1, 2))
    
    def _get_specific_bcs(self, f):
        return [1, 1] # Periodic


class BurgersDataset(BaseHDF5DirectoryDataset):
    """
    Order Vx, Vy, density, pressure
    """
    @staticmethod
    def _specifics():
        time_index = 1
        sample_index = 0
        field_names = ['Vx']
        type = 'burgers'
        split_level = 'sample'
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f):
        samples = f['tensor'].shape[0] 
        steps = f['tensor'].shape[1]# Per dset
        return samples, steps

    def _reconstruct_file(self, file, sample_idx):
        vx = file['tensor'][sample_idx, :]
        # print(vx.shape)
        vx = vx[:, None, :]
        return vx#.transpose((0, 3, 1, 2))
    
    def _reconstruct_sample(self, file, sample_idx, time_idx, n_steps):
        vx = file['tensor'][sample_idx, time_idx-n_steps*self.dt:time_idx+self.dt]
        # print(vx.shape)
        vx = vx[:, None, :]
        return vx#.transpose((0, 3, 1, 2))
    
    def _get_specific_bcs(self, f):
        return [1] # Periodic


PDEBENCH_NAME_TO_OBJECT = {
    'swe': SWEDataset,
    'incompNS': IncompNSDataset,
    'diffre2d': DiffRe2DDataset,
    'compNS': CompNSDataset,
    'burgers': BurgersDataset,
}


class NPYDataset(Dataset):
    """
    Dataset loader for custom NPY files with multiple format support.
    
    Supports:
    - Single file: [N, resolution, resolution, time]
    - Multiple files: auto-concatenated [N, resolution, resolution, time]
    - Multi-field files: data_0_u.npy, data_0_v.npy, etc. (auto-stacked as channels)
    
    Args:
        path (str): Path to directory containing .npy files or single .npy file
        resolution (tuple): Target resolution for spatial dimensions (H, W) or (D, H, W)
        in_channels (int): Number of channels/fields in the data
        n_steps (int): Number of timesteps per sample
        dt (int): Temporal stride
        split (str): 'train', 'valid', or 'test'
        train_val_test (tuple): Proportion of data to use for each split
        subname (str): Name override for the dataset
        field_suffixes (list): List of field suffixes to load, e.g., ['_u', '_v']. 
                              If None, auto-detect from files.
    """
    def __init__(self, path, resolution, in_channels=1, n_steps=1, dt=1, split='train',
                 train_val_test=None, subname=None, field_suffixes=None):
        super().__init__()
        self.path = path
        self.resolution = resolution if isinstance(resolution, tuple) else tuple(resolution)
        self.in_channels = in_channels
        self.n_steps = n_steps
        self.dt = dt
        self.split = split
        self.train_val_test = train_val_test if train_val_test else (0.8, 0.1, 0.1)
        self.partition = {'train': 0, 'valid': 1, 'test': 2}[split]
        self.field_suffixes = field_suffixes
        
        if subname is None:
            self.subname = os.path.basename(path)
        else:
            self.subname = subname
        
        self.dataset_name = self.subname
        self.field_names = [f'field_{i}' for i in range(in_channels)]
        self.boundary_conditions = 'periodic'  # Default; can be overridden
        self.lazy_mode = False
        self._array_cache = {}
        self._sample_map = []
        self._singlefield_files = []
        self._multifield_groups = []
        self._source_files_for_logging = []
        self._n_time = None
        
        # Load data
        self._load_data()

    def _get_cached_array(self, filepath):
        """Load npy as mmap and cache handle."""
        arr = self._array_cache.get(filepath)
        if arr is None:
            arr = np.load(filepath, mmap_mode='r')
            self._array_cache[filepath] = arr
        return arr

    def _split_sample_indices(self, total_samples):
        """Build split indices [start, end) for train/valid/test."""
        sample_per_part = np.ceil(np.array(self.train_val_test) * total_samples).astype(int)
        sample_per_part[2] = max(total_samples - sample_per_part[0] - sample_per_part[1], 0)
        starts = np.cumsum([0] + sample_per_part.tolist())
        return starts[self.partition], starts[self.partition + 1]

    def _build_lazy_singlefield(self, files, use_full_split=False):
        """Build lazy metadata for single-field directory dataset."""
        file_infos = []
        base_spatial = None
        detected_time_axis = None
        for f in files:
            arr = self._get_cached_array(f)
            if arr.ndim != 4:
                # Not a 4D data file — likely something else (masks should already be excluded).
                print(f"WARNING: Skipping non-4D file when building single-field dataset {f}: {arr.shape}")
                continue

            # Detect whether ordering is [N, H, W, T] or [N, T, H, W]
            if tuple(arr.shape[1:3]) == tuple(self.resolution):
                time_ax = 3
                spatial = tuple(arr.shape[1:3])
            elif tuple(arr.shape[2:4]) == tuple(self.resolution):
                time_ax = 1
                spatial = tuple(arr.shape[2:4])
            else:
                # Fallback: assume last axis is time
                time_ax = 3
                spatial = tuple(arr.shape[1:3])

            if base_spatial is None:
                base_spatial = spatial
                detected_time_axis = time_ax
            elif spatial != base_spatial:
                print(f"WARNING: Dropping file with incompatible spatial shape {arr.shape} in {self.path}")
                continue

            n_time = arr.shape[time_ax]
            file_infos.append({'file': f, 'n_samples': arr.shape[0], 'n_time': n_time, 'time_axis': detected_time_axis})

        if not file_infos:
            raise ValueError(f"No shape-compatible .npy files found in {self.path}")

        self._singlefield_files = file_infos
        self._source_files_for_logging = [fi['file'] for fi in file_infos]
        self.in_channels = 1
        self._n_time = file_infos[0]['n_time']
        self._single_time_axis = file_infos[0].get('time_axis', 3)
        for fi in file_infos:
            if fi['n_time'] != self._n_time:
                raise ValueError("All files must have identical time dimension for lazy loading")

        total_samples = sum(fi['n_samples'] for fi in file_infos)
        if use_full_split:
            start_idx, end_idx = 0, total_samples
        else:
            start_idx, end_idx = self._split_sample_indices(total_samples)

        self._sample_map = []
        global_offset = 0
        for file_idx, fi in enumerate(file_infos):
            file_start = global_offset
            file_end = global_offset + fi['n_samples']
            overlap_start = max(file_start, start_idx)
            overlap_end = min(file_end, end_idx)
            if overlap_start < overlap_end:
                local_start = overlap_start - file_start
                local_end = overlap_end - file_start
                for local_idx in range(local_start, local_end):
                    self._sample_map.append((file_idx, local_idx))
            global_offset = file_end

    def _build_lazy_multifield(self, base_names, suffixes, use_full_split=False):
        """Build lazy metadata for multi-field directory dataset."""
        groups = []
        for base_name in sorted(base_names.keys()):
            suffix_to_file = {suf: fp for suf, fp in base_names[base_name]}

            # Choose channel order: prefer velocity_x, velocity_y, pressure
            desired_channels = [
                ['velocity_x', 'velx', 'vx', 'u'],
                ['velocity_y', 'vely', 'vy', 'v'],
                ['pressure', 'p']
            ]

            channels = []  # list of (matched_suffix_key, filepath)
            for token_list in desired_channels:
                matched = None
                for key in suffix_to_file.keys():
                    for token in token_list:
                        if token in key:
                            matched = key
                            break
                    if matched is not None:
                        break
                if matched is not None:
                    channels.append((matched, suffix_to_file[matched]))

            # If no core channels found, skip this base_name
            if len(channels) == 0:
                continue

            # Validate shapes using the first channel found, accepting either [N, H, W, T] or [N, T, H, W]
            first_arr = self._get_cached_array(channels[0][1])
            if first_arr.ndim != 4:
                raise ValueError(f"Unexpected multi-field shape in {channels[0][1]}: {first_arr.shape}. Expected 4D array")
            n_samples = first_arr.shape[0]
            # Determine time axis by matching spatial resolution
            if tuple(first_arr.shape[1:3]) == tuple(self.resolution):
                time_ax = 3
                spatial = tuple(first_arr.shape[1:3])
            elif tuple(first_arr.shape[2:4]) == tuple(self.resolution):
                time_ax = 1
                spatial = tuple(first_arr.shape[2:4])
            else:
                # Fallback to last axis as time
                time_ax = 3
                spatial = tuple(first_arr.shape[1:3])

            n_time = first_arr.shape[time_ax]

            # Validate remaining chosen channels for consistency
            for _, filepath in channels[1:]:
                arr = self._get_cached_array(filepath)
                if arr.ndim != 4 or arr.shape[0] != n_samples or arr.shape[time_ax] != n_time:
                    raise ValueError(f"Inconsistent multi-field array shape for base {base_name}")

            # Optional mask file
            mask_file = None
            for key, fp in suffix_to_file.items():
                if 'mask' in key:
                    mask_file = fp
                    break

            groups.append({
                'base_name': base_name,
                'files': suffix_to_file,
                'channels': channels,
                'n_samples': n_samples,
                'n_time': n_time,
                'time_axis': time_ax,
                'mask_file': mask_file,
            })

        if not groups:
            raise ValueError(f"Could not load multi-field data from {self.path}")

        self._multifield_groups = groups
        self._source_files_for_logging = [fp for g in groups for fp in g['files'].values()]
        self.in_channels = len(suffixes)
        self._n_time = groups[0]['n_time']
        for g in groups:
            if g['n_time'] != self._n_time:
                raise ValueError("All multi-field groups must have identical time dimension for lazy loading")

        total_samples = sum(g['n_samples'] for g in groups)
        if use_full_split:
            start_idx, end_idx = 0, total_samples
        else:
            start_idx, end_idx = self._split_sample_indices(total_samples)

        self._sample_map = []
        global_offset = 0
        for group_idx, group in enumerate(groups):
            group_start = global_offset
            group_end = global_offset + group['n_samples']
            overlap_start = max(group_start, start_idx)
            overlap_end = min(group_end, end_idx)
            if overlap_start < overlap_end:
                local_start = overlap_start - group_start
                local_end = overlap_end - group_start
                for local_idx in range(local_start, local_end):
                    self._sample_map.append((group_idx, local_idx))
            global_offset = group_end
    
    def _detect_field_patterns(self, npy_files):
        """Detect multi-field patterns like data_0_u.npy, data_0_v.npy"""
        if len(npy_files) <= 1:
            return None, None
        
        # Extract base names and suffixes. Accept a broader set of suffixes
        suffixes = {}
        base_names = {}

        known_suffix_tokens = [
            '_u', '_v', '_w', '_p', '_rho', '_T',
            '_component0', '_component1', '_component2',
            '_velocity_x', '_velocity_y', '_velocity_z', '_pressure', '_mask',
            '_vorticity', '_wallclock', '_errors', '_filelist', '_velx', '_vely', '_velz'
        ]

        for f in npy_files:
            filename = os.path.basename(f)
            matched = False
            # Try to find one of the known suffix tokens anywhere in the filename
            for suffix in known_suffix_tokens:
                if suffix in filename:
                    base = filename.replace(suffix + '.npy', '')
                    if base not in base_names:
                        base_names[base] = []
                    base_names[base].append((suffix, f))
                    suffixes[suffix] = True
                    matched = True
                    break
            if not matched:
                # Fallback: attempt to split by last underscore to find base/suffix
                if '_' in filename:
                    idx = filename.rfind('_')
                    base = filename[:idx]
                    suffix = filename[idx:len(filename) - 4]  # remove .npy
                    base_names.setdefault(base, []).append((f'_{suffix}', f))
                    suffixes[f'_{suffix}'] = True
        
        has_multifield_base = any(len(entries) > 1 for entries in base_names.values())
        if len(suffixes) > 0 and has_multifield_base:
            # Multi-field pattern detected
            detected_suffixes = sorted(suffixes.keys())
            return detected_suffixes, base_names
        
        return None, None
        
    def _load_multifield_data(self, base_names, suffixes):
        """Load multi-field files and stack them as channels."""
        data_list = []
        
        # Sort base names to ensure consistent ordering
        for base_name in sorted(base_names.keys()):
            field_data_list = []
            
            # Load each field in order
            for suffix in suffixes:
                # Find the file with this suffix for this base name
                for suf, filepath in base_names[base_name]:
                    if suf == suffix:
                        field_data = np.load(filepath)
                        # Expected: [N, H, W, T]
                        field_data_list.append(field_data)
                        break
            
            if field_data_list:
                # Stack fields along channel dimension
                # Input: [N, H, W, T], Output: [N, H, W, fields, T]
                stacked = np.stack(field_data_list, axis=3)  # [N, H, W, fields, T]
                data_list.append(stacked)
        
        if data_list:
            # Concatenate all samples
            all_data = np.concatenate(data_list, axis=0)
            return all_data
        else:
            raise ValueError(f"Could not load multi-field data from {self.path}")

    def _load_singlefield_data(self, files):
        """Load single-field files and concatenate only shape-compatible arrays."""
        arrays = [np.load(f) for f in files]
        if len(arrays) == 1:
            return arrays[0]

        # Match on non-batch dimensions only; batch dimension may differ and is concat axis
        shape_counts = {}
        for arr in arrays:
            key = tuple(arr.shape[1:])
            shape_counts[key] = shape_counts.get(key, 0) + 1

        target_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
        filtered = [arr for arr in arrays if tuple(arr.shape[1:]) == target_shape]
        dropped = len(arrays) - len(filtered)

        if dropped > 0:
            print(
                f"WARNING: Dropping {dropped} file(s) with incompatible shape in {self.path}. "
                f"Using shape (*, {', '.join(map(str, target_shape))})."
            )

        if not filtered:
            raise ValueError(f"No shape-compatible .npy files found in {self.path}")

        return np.concatenate(filtered, axis=0)
        
    def _load_data(self):
        """Load NPY files from path. Auto-detects _val.npy files for validation/test splits."""
        # Flag to track if data was loaded from validation files
        self.loaded_from_val = False
        
        # Check if path is a file or directory
        if os.path.isfile(self.path):
            # Single file case - check for corresponding _val file
            self.data = np.load(self.path)
            self.npy_files = [self.path]
            
            # Look for corresponding _val file
            base_path = self.path.replace('.npy', '')
            val_path = base_path + '_val.npy'
            
            if os.path.isfile(val_path) and self.split in ['valid', 'test']:
                print(f"Loading validation/test data from {val_path}")
                self.data = np.load(val_path)
                self.loaded_from_val = True
            
            detected_suffixes = None
            # Attempt to find a sibling mask file (e.g., basename_mask.npy)
            base_path = self.path.replace('.npy', '')
            mask_candidates = [base_path + '_mask.npy', base_path + 'mask.npy']
            mask_found = None
            for mc in mask_candidates:
                if os.path.isfile(mc):
                    mask_found = mc
                    break
            # also search same directory for any file containing 'mask' with same base prefix
            if mask_found is None:
                dirp = os.path.dirname(self.path)
                bn = os.path.basename(base_path)
                for f in sorted(glob.glob(os.path.join(dirp, '*mask*.npy'))):
                    if bn in os.path.basename(f):
                        mask_found = f
                        break
            if mask_found is not None:
                try:
                    m = np.load(mask_found)
                    # Expect mask shape [N, H, W] or [H, W]
                    if m.ndim == 3:
                        self._mask_data = m
                    elif m.ndim == 2:
                        # broadcast to samples
                        n_samples = self.data.shape[0] if self.data.ndim >= 4 else 1
                        self._mask_data = np.stack([m] * n_samples, axis=0)
                    else:
                        # unexpected, ignore
                        self._mask_data = None
                except Exception:
                    self._mask_data = None
            else:
                self._mask_data = None
        else:
            # Directory case - lazy load to avoid keeping all data in RAM
            all_npy_files = sorted(glob.glob(os.path.join(self.path, '*.npy')))
            
            # Separate split-specific files when they exist.
            # Breakflow stores train trajectories in data_0_* files and test
            # trajectories in data_test_* files.
            train_files = [f for f in all_npy_files if '_val' not in os.path.basename(f) and '_test' not in os.path.basename(f)]
            val_files = [f for f in all_npy_files if '_val' in os.path.basename(f)]
            test_files = [f for f in all_npy_files if '_test' in os.path.basename(f)]
            
            if not train_files and not val_files and not test_files:
                raise FileNotFoundError(f"No .npy files found in {self.path}")
            
            # Prefer explicit test files for the test split, then explicit
            # validation files for validation, otherwise fall back to the
            # training files and split them by sample index.
            if self.split == 'test' and test_files:
                print(f"Using {len(test_files)} dedicated test files for test split")
                split_files = test_files
                use_full_split = True
            elif self.split == 'valid' and val_files:
                print(f"Using {len(val_files)} dedicated validation files for valid split")
                split_files = val_files
                use_full_split = True
            elif self.split == 'test' and val_files:
                print(f"Using {len(val_files)} validation files for test split")
                split_files = val_files
                use_full_split = True
            else:
                split_files = train_files if train_files else all_npy_files
                use_full_split = False

            if not split_files:
                raise FileNotFoundError(f"No usable .npy files found in {self.path} for split={self.split}")

            # Use dedicated split files if available.
            if use_full_split:
                # Detect if validation files have multi-field pattern
                detected_suffixes, base_names = self._detect_field_patterns(split_files)
                
                if detected_suffixes and base_names:
                    print(f"Detected multi-field pattern in {self.split} data with suffixes: {detected_suffixes}")
                    self.lazy_mode = True
                    self._build_lazy_multifield(base_names, detected_suffixes, use_full_split=True)
                    self.npy_files = split_files
                else:
                    # Single-field validation dataset
                    self.lazy_mode = True
                    self._build_lazy_singlefield(split_files, use_full_split=True)
                    self.npy_files = split_files
                
                self.loaded_from_val = True
            else:
                # Use training files
                detected_suffixes, base_names = self._detect_field_patterns(split_files)
                
                if detected_suffixes and base_names:
                    # Multi-field dataset (like Gray Scott)
                    print(f"Detected multi-field pattern with suffixes: {detected_suffixes}")
                    self.lazy_mode = True
                    self._build_lazy_multifield(base_names, detected_suffixes)
                    self.npy_files = split_files
                else:
                    # Single-field dataset - load and concatenate all files
                    self.lazy_mode = True
                    self._build_lazy_singlefield(split_files)
                    self.npy_files = split_files
                
                self.loaded_from_val = False

        if self.lazy_mode:
            self.field_names = [f'field_{i}' for i in range(self.in_channels)]
            n_time = self._n_time
            n_steps_total = self.n_steps + self.dt - 1
            if n_time <= n_steps_total:
                raise ValueError(f"Data has {n_time} timesteps but n_steps={self.n_steps} requires {n_steps_total}")
            self.n_valid_times = n_time - n_steps_total
            self.len = len(self._sample_map) * self.n_valid_times
            return
        
        # Expected format: [N, resolution, resolution, time] or [N, H, W, channels, T]
        # Reshape to [N, time, channels, resolution, resolution]
        if self.data.ndim == 4:
            # Could be [N, H, W, T] or [N, T, H, W]. Detect spatial axes by resolution.
            if tuple(self.data.shape[1:3]) == tuple(self.resolution):
                # [N, H, W, T] -> [N, T, H, W]
                self.data = np.transpose(self.data, (0, 3, 1, 2))
            else:
                # assume already [N, T, H, W]
                pass
            # Add channel axis: [N, T, 1, H, W]
            self.data = self.data[:, :, np.newaxis, :, :]
        elif self.data.ndim == 5:
            # Could be [N, H, W, C, T] or [N, T, C, H, W]
            # Check if axis 1 looks like time (should be largest typical value like 50)
            # and axis 4 is small (like 64 for resolution)
            if self.data.shape[1] > self.data.shape[3]:
                # Likely [N, T, C, H, W] - already correct
                pass
            else:
                # Likely [N, H, W, C, T] - transpose to [N, T, C, H, W]
                self.data = np.transpose(self.data, (0, 4, 3, 1, 2))
        
        # Verify data shape
        if self.data.ndim != 5:
            raise ValueError(f"Unexpected data shape: {self.data.shape}. Expected [N, T, C, H, W]")
        
        # Update field names based on actual channels
        self.field_names = [f'field_{i}' for i in range(self.data.shape[2])]
        
        # Split data according to train_val_test
        n_samples = self.data.shape[0]
        
        # If data was loaded from validation files, use all of it without further splitting
        if self.loaded_from_val:
            self.data_split = self.data
        else:
            # For training data, apply train_val_test split
            sample_per_part = np.ceil(np.array(self.train_val_test) * n_samples).astype(int)
            sample_per_part[2] = max(n_samples - sample_per_part[0] - sample_per_part[1], 0)
            
            starts = np.cumsum([0] + sample_per_part.tolist())
            self.data_split = self.data[starts[self.partition]:starts[self.partition+1]]
        
        # Calculate valid indices
        n_time = self.data_split.shape[1]
        n_steps_total = self.n_steps + self.dt - 1
        if n_time <= n_steps_total:
            raise ValueError(f"Data has {n_time} timesteps but n_steps={self.n_steps} requires {n_steps_total}")
        
        self.n_valid_times = n_time - n_steps_total
        self.len = self.data_split.shape[0] * self.n_valid_times
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Compute which sample and time index
        sample_idx = index // self.n_valid_times
        time_idx = index % self.n_valid_times

        if self.lazy_mode:
            if self._multifield_groups:
                group_idx, local_sample_idx = self._sample_map[sample_idx]
                group = self._multifield_groups[group_idx]
                field_arrays = []
                # Load channels in the selected order and normalize per-sample to [T, H, W]
                for matched_suffix, filepath in group.get('channels', []):
                    arr = self._get_cached_array(filepath)
                    sample_arr = arr[local_sample_idx]  # could be [H, W, T] or [T, H, W]
                    # Ensure sample_arr is 3D
                    if sample_arr.ndim != 3:
                        sample_arr = np.squeeze(sample_arr)
                    # Normalize to [T, H, W]
                    if sample_arr.shape[0] == group['n_time']:
                        norm = sample_arr
                    elif sample_arr.shape[-1] == group['n_time']:
                        norm = np.transpose(sample_arr, (2, 0, 1))
                    else:
                        # Fallback heuristics: prefer time-first if single_time_axis indicates so
                        if group.get('time_axis', 3) == 1 and sample_arr.shape[0] > sample_arr.shape[-1]:
                            norm = sample_arr
                        else:
                            norm = np.transpose(sample_arr, (2, 0, 1))
                    field_arrays.append(norm)
                # Stack into [T, C, H, W]
                stacked = np.stack(field_arrays, axis=1)
                sample_data = stacked  # [T, C, H, W]
                file_name = group['base_name']
                # Load mask if present in group
                mask = None
                if 'mask_file' in group and group['mask_file'] is not None:
                    mask_arr = self._get_cached_array(group['mask_file'])
                    # mask_arr may be [N, H, W] or [H, W]
                    if mask_arr.ndim == 3:
                        mask = mask_arr[local_sample_idx]
                    elif mask_arr.ndim == 2:
                        mask = mask_arr
                    else:
                        # if mask has time dim accidentally, take first time
                        if mask_arr.ndim == 4:
                            mask = mask_arr[local_sample_idx, ..., 0]
                        else:
                            mask = mask_arr
            else:
                file_idx, local_sample_idx = self._sample_map[sample_idx]
                file_info = self._singlefield_files[file_idx]
                arr = self._get_cached_array(file_info['file'])
                sample = arr[local_sample_idx]
                if sample.ndim != 3:
                    sample = np.squeeze(sample)
                # Normalize to [T, H, W]
                if sample.shape[0] == self._n_time:
                    sample_norm = sample
                elif sample.shape[-1] == self._n_time:
                    sample_norm = np.transpose(sample, (2, 0, 1))
                else:
                    if getattr(self, '_single_time_axis', 3) == 1:
                        sample_norm = sample
                    else:
                        sample_norm = np.transpose(sample, (2, 0, 1))
                sample_data = sample_norm[:, np.newaxis, :, :]  # [T, 1, H, W]
                file_name = os.path.basename(file_info['file'])

            trajectory = sample_data[time_idx:time_idx+self.n_steps+self.dt:self.dt]
            input_traj = trajectory[:-1]
            output_traj = trajectory[[-1]]

            input_fields = subsample(torch.from_numpy(input_traj).float(), self.resolution)
            output_fields = subsample(torch.from_numpy(output_traj).float(), self.resolution)

            out = {
                'input_fields': input_fields,
                'output_fields': output_fields,
                'boundary_conditions': torch.as_tensor([1.0]),  # Assuming periodic
                'name': self.dataset_name,
                'file': file_name,
                'index': torch.as_tensor(index),
            }
            if 'mask' in locals() and mask is not None:
                out['mask'] = torch.from_numpy(mask).float()
            return out
        
        # Extract trajectory [n_steps, channels, res, res]
        trajectory = self.data_split[sample_idx, time_idx:time_idx+self.n_steps+self.dt:self.dt]
        
        # Subsample to target resolution if needed
        input_traj = trajectory[:-1]  # All but last timestep
        output_traj = trajectory[[-1]]  # Last timestep
        
        input_fields = subsample(torch.from_numpy(input_traj).float(), self.resolution)
        output_fields = subsample(torch.from_numpy(output_traj).float(), self.resolution)
        
        return {
            'input_fields': input_fields,
            'output_fields': output_fields,
            'boundary_conditions': torch.as_tensor([1.0]),  # Assuming periodic
            'name': self.dataset_name,
            'file': self.npy_files[0].split('/')[-1] if len(self.npy_files) == 1 else 'concatenated',
            'index': torch.as_tensor(index),
        }
        # Attach mask if available for non-lazy single-file datasets
        if hasattr(self, '_mask_data') and self._mask_data is not None:
            # sample_idx is index // self.n_valid_times
            mask_sample = self._mask_data[sample_idx]
            mask_t = subsample(torch.from_numpy(mask_sample).float(), self.resolution)
            # mask_t shape [H,W] -> make [1,1,H,W] then batch dim will be added by collate
            # Return mask as [H,W] per-sample to be collated
            out['mask'] = mask_t
        return out
    
    def get_name(self, full_name=False):
        if full_name:
            return self.subname
        return self.subname


class MixedDataset(Dataset):
    def __init__(
        self, dataset_names=[], n_past=1, n_future=1, dt=1, train_val_test=(.8, .1, .1),
        split='train', extended_names=False, train_offset=0
    ):
        super().__init__()
        # Global dicts used by Mixed DSET. 
        self.train_offset = train_offset
        self.path_list = [DATASET_SPECS[name]['main_path'] for name in dataset_names]
        # self.type_list = [('compNS' if name[:6] == 'compNS' in name else name) for name in dataset_names]
        self.include_string = [DATASET_SPECS[name]['include_string'] for name in dataset_names]
        # self.reso = [[int(d) for d in tuple(str(res).split('x'))] for res in self.reso]
        self.reso = [DATASET_SPECS[name]['resolution'] for name in dataset_names]

        self.extended_names = extended_names
        self.split = split
        self.sub_dsets = []
        self.offsets = [0]
        self.train_val_test = train_val_test

        for dset_name, path, include_string, reso in zip(dataset_names, self.path_list, self.include_string, self.reso):
            object_type = 'compNS' if dset_name[:6] == 'compNS' else dset_name
            
            # Custom NPY datasets (group == 'custom_npy')
            if DATASET_SPECS.get(dset_name, {}).get('group') == 'custom_npy':
                in_channels = DATASET_SPECS[dset_name].get('in_channels', 1)
                if dset_name == 'breakflow':
                    # Breakflow has different trajectory lengths for train/valid
                    # versus test. Train/valid files are 40 steps long, while the
                    # dedicated test files are 200 steps long.
                    max_steps = DATASET_SPECS[dset_name].get(
                        'test_n_steps' if split == 'test' else 'train_n_steps',
                        DATASET_SPECS[dset_name]['n_steps'],
                    )
                else:
                    max_steps = DATASET_SPECS[dset_name]['n_steps']

                n_steps_adjusted = min(n_past + n_future - 1, max_steps - 1)
                subdset = NPYDataset(
                    path, reso, in_channels=in_channels, n_steps=n_steps_adjusted, dt=dt,
                    train_val_test=train_val_test, split=split, subname=dset_name
                )
            elif object_type in PDEBENCH_NAME_TO_OBJECT:  # PDEBench dataset
                n_steps_adjusted = min(n_past+n_future-1, DATASET_SPECS[dset_name]['n_steps']-1)  # nb of steps in a context (past)
                subdset = PDEBENCH_NAME_TO_OBJECT[object_type](
                    path, reso, include_string, n_steps=n_steps_adjusted, dt=dt, train_val_test=train_val_test, split=split
                )
            else:  # the-well dataset
                subdset = GenericWellDataset(
                    path=path,
                    # well_dataset_name=dset_name,
                    well_split_name=split,
                    resolution=reso,
                    include_filters=include_string,
                    exclude_filters=[],
                    n_steps_input=n_past,
                    n_steps_output=n_future,
                    dt_stride=dt,
                    name_override=dset_name,
                )
            # Check to make sure our dataset actually exists with these settings
            try:
                len(subdset)
            except ValueError:
                raise ValueError(f'Dataset {path} is empty. Check that n_steps < trajectory_length in file.')
            self.sub_dsets.append(subdset)
            self.offsets.append(self.offsets[-1]+len(self.sub_dsets[-1]))
        self.offsets[0] = -1

        self.subset_dict = self._build_subset_dict()

    def get_state_names(self):
        name_list = []
        visited = set()
        for dset in self.sub_dsets:
            name = dset.get_name() # Could use extended names here
            if not name in visited:
                visited.add(name)
                name_list.append(dset.field_names)
        return [f for fl in name_list for f in fl] # Flatten the names

    def _build_subset_dict(self):
        """ Maps fields to subsets of variables """
        subdset = self.sub_dsets[0]
        dset_name = subdset.get_name()

        def _num_channels(dset):
            if hasattr(dset, 'in_channels') and dset.in_channels is not None:
                return int(dset.in_channels)
            if hasattr(dset, 'data'):
                return int(dset.data.shape[2])
            raise AttributeError(f"Dataset {dset.get_name()} has neither in_channels nor data")
        # Handle all custom NPY datasets
        # Accept either literal keys or dataset entries whose spec group is 'custom_npy'
        if DATASET_SPECS.get(dset_name, {}).get('group') == 'custom_npy' or \
           dset_name in ['custom_npy', 'navierstokes', 'kuramotosivashinsky', 'grayscott']:
            # For custom NPY datasets, create field labels based on actual data channels
            subset_dict = {}
            for dataset in self.sub_dsets:
                name = dataset.get_name()
                n_ch = _num_channels(dataset)
                subset_dict[name] = list(range(n_ch))
            return subset_dict
        elif dset_name in PDEBENCH_NAME_TO_OBJECT: # PDEBench dataset
            cur_max = 0
            subset_dict = {}
            for name, dset in PDEBENCH_NAME_TO_OBJECT.items():
                field_names = dset._specifics()[2]
                subset_dict[name] = list(range(cur_max, cur_max + len(field_names)))
                cur_max += len(field_names)
            return subset_dict
        else: # the-well dataset
            return {
                'euler_pbc': [0,1,2,3,4],                # density, energy, pressure, mom_x, mom_y
                'euler_obc': [0,1,2,3,4],                # density, energy, pressure, mom_x, mom_y
                'shear_flow': [5,2,6,7],                 # tracer, pressure, vel_x, vel_y
                'active_matter': [8,6,7,9,10,11,12,25,26,27,28], # concentration, vel_x, vel_y, tsr_xx, tsr_xy, tsr_yx, tsr_yy, tsr_xx, tsr_xy, tsr_yx, tsr_yy,
                'gray_scott': [13,14],                   # concentration_1, concentration_2
                'helmholtz': [15,16],                    # pressure_real, pressure_imag
                'rayleigh_benard': [17,2,6,7],           # buoyancy, pressure, vel_x, vel_y
                'rayleigh_taylor': [0,18,19,20],         # density, vel_x, vel_y, vel_z
                'supernova': [2,0,21,18,19,20],          # pressure, density, temperature, vel_x, vel_y, vel_z
                'mhd': [0,18,19,20,22,23,24],            # density, vel_x, vel_y, vel_z, mag_x, mag_y, mag_z
                'gravity_cooling': [2,0,21,18,19,20],    # pressure, density, temperature, vel_x, vel_y, vel_z
                'radiative_layer': [0,2,6,7],            # density, pressure, vel_x, vel_y
                'convective_envelope': [1,0,2,18,19,20], # energy, density, pressure, vel_x, vel_y, vel_z
                # finetuning datasets
                'euler_obc_finetune': [0,1,2,3,4],       # density, energy, pressure, mom_x, mom_y
            }
            
    def __getitem__(self, index):
        file_idx = np.searchsorted(self.offsets, index, side='right')-1 #which dataset are we on
        local_idx = index - max(self.offsets[file_idx], 0)
        try:
            dict = self.sub_dsets[file_idx][local_idx]
        except:
            print('FAILED AT ', file_idx, local_idx, index,int(os.environ.get("RANK", 0)))
            thisvariabledoesntexist
        # return x, file_idx, torch.tensor(self.subset_dict[self.sub_dsets[file_idx].get_name()]), bcs, y
        dataset = self.sub_dsets[file_idx]
        dict['field_labels'] = torch.tensor(self.subset_dict[dataset.get_name()])
        dict['file_index'] = file_idx
        return dict
    
    def __len__(self):
        return sum([len(dset) for dset in self.sub_dsets])


class RandomSamplerSeed(Sampler[int]):
    """ Overwrite the RandomSampler to allow for a seed for each epoch.
    Effectively going over the same data at same epochs. """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, 
                 epoch: Optional[int] = None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.epoch = epoch

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=g).item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


T_co = TypeVar('T_co', covariant=True)


class MultisetSampler(Sampler[T_co]):
    """ Sampler that restricts data loading to a subset of the dataset. """

    def __init__(self, 
        dataset: MixedDataset, batch_size: int, 
        shuffle: bool = True, seed: int = 0,
        max_samples: int = 10, rank: int = 0, 
        distributed: bool = True,
    ):
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        if distributed:
            sampler = DistributedSampler
        else:
            sampler = RandomSamplerSeed
        self.sub_samplers = [sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[T_co]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        generator = torch.Generator()
        generator.manual_seed(100*self.epoch+10*self.seed+self.rank)
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = torch.randint(0, len(sampler_choices), size=(1,), generator=generator).item()
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            # Do drop last batch type logic - if you can get a full batch, yield it, otherwise move to next dataset
            try:
                queue = []
                for _ in range(self.batch_size):
                    queue.append(next(samplers[dset_sampled]) + offset)
                if len(queue) == self.batch_size:
                    for d in queue:
                        yield d
            except Exception as err:
                print('ERRRR', err)
                sampler_choices.pop(index_sampled)
                print(f'Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}')
                continue
            if count >= self.max_samples:
                break
    
    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """ Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)
        self.epoch = epoch


def get_data_objects(
    dataset_names: List[str],
    batch_size: int,
    epoch_size: int,
    train_val_test: Tuple[float],
    n_past: int, 
    n_future: int,
    distributed: bool, 
    num_data_workers: int,
    rank: int, 
    split: str,
) -> Tuple:
    
    dataset = MixedDataset(
        dataset_names, n_past=n_past, n_future=n_future, 
        train_val_test=train_val_test, split=split,
        train_offset=0
    )

    sampler = MultisetSampler(  # default: shuffle = True 
        dataset, batch_size, distributed=distributed, 
        max_samples=epoch_size, rank=rank
    )

    dataloader = DataLoader(
        dataset, batch_size=int(batch_size), 
        num_workers=num_data_workers,
        shuffle=False, #(sampler is None), # shuffle determined by the sampler
        sampler=sampler, # Since validation is on a subset, use a fixed random subset,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, sampler, dataloader
