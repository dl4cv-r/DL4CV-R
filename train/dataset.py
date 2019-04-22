import h5py
from torch.utils.data import Dataset
from pathlib import Path
from random import choice
import numpy as np


class TrainingDataset(Dataset):
    """
    New Dataset for new data API.
    Loads downsampled images from HDF5 files.
    This makes a lot of presumptions about how the data is structured in folders.
    """
    def __init__(self, data_root, transform, single_coil=False, acc_fac=None):

        super().__init__()

        if acc_fac is not None:  # Use both if self.acc_fac is None.
            assert acc_fac in (4, 8), 'Invalid acceleration factor'

        self.data_root = data_root
        self.transform = transform
        self.acc_fac = acc_fac
        self.recons_key = 'reconstruction_esc' if single_coil else 'reconstruction_rss'

        data_path = Path(self.data_root)

        print(f'Initializing {data_path.stem}. This might take a minute.')

        if self.acc_fac:
            data_path_ = data_path / str(self.acc_fac)
            data_paths = list(data_path_.glob('*.h5'))
            data_paths.sort()
        else:
            data_path_4 = data_path / '4'
            data_path_8 = data_path / '8'
            data_paths = list(data_path_4.glob('*.h5')) + list(data_path_8.glob('*.h5'))
            data_paths.sort()

        label_path = data_path / self.recons_key
        label_paths = [str(file) for file in label_path.glob('*.h5')]
        label_paths.sort()

        if not (data_paths and label_paths):  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        # Just train for every file, whether 4-fold or 8-fold. Too annoying to separate them.
        slice_counts = [self.get_slice_number(str(file_name)) for file_name in data_paths]
        self.num_slices = sum(slice_counts)

        paths_and_slices = list()
        for path, slice_num in zip(data_paths, slice_counts):
            paths_and_slices += [[path, s_idx] for s_idx in range(slice_num)]

        self.paths_and_slices = paths_and_slices
        assert self.num_slices == len(paths_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} initialization!')

    def __len__(self):
        return self.num_slices

    @staticmethod
    def get_slice_number(file_name):  # For data files only, not label files.
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            return hf['data'].shape[0]

    def slice_parse_fn(self, data_path, slice_num):
        assert isinstance(data_path, Path) and isinstance(slice_num, int)

        with h5py.File(data_path, 'r', swmr=True) as dhf:
            ds_slice_arr = np.asarray(dhf['data'][slice_num])

            fat = dhf.attrs['acquisition']
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')

        label_path = data_path.parents[1] / self.recons_key / data_path.name  # Assumes data folder structure.
        with h5py.File(label_path, 'r', swmr=True) as lhf:
            gt_slice_arr = np.asarray(lhf[self.recons_key][slice_num])

        return ds_slice_arr, gt_slice_arr, fat_supp

    def __getitem__(self, idx):
        file_path, slice_num = self.paths_and_slices[idx]
        ds_slice, gt_slice, fat_supp = self.slice_parse_fn(file_path, slice_num)
        data, labels = self.transform(ds_slice, gt_slice, fat_supp)
        return data, labels


class SubmissionDataset(Dataset):
    """
    New Dataset for new data API.
    Loads downsampled images from HDF5 files.
    This makes a lot of presumptions about how the data is structured in folders.
    """
    def __init__(self, data_root, transform, single_coil=False, acc_fac=None):

        super().__init__()

        if acc_fac is not None:  # Use both if self.acc_fac is None.
            assert acc_fac in (4, 8), 'Invalid acceleration factor'

        self.data_root = data_root
        self.transform = transform
        self.acc_fac = acc_fac
        self.recons_key = 'reconstruction_esc' if single_coil else 'reconstruction_rss'

        data_path = Path(self.data_root)

        print(f'Initializing {data_path.stem}. This might take a minute.')

        if self.acc_fac:
            data_path_ = data_path / str(self.acc_fac)
            data_paths = list(data_path_.glob('*.h5'))
            data_paths.sort()
        else:
            data_path_4 = data_path / '4'
            data_path_8 = data_path / '8'
            data_paths = list(data_path_4.glob('*.h5')) + list(data_path_8.glob('*.h5'))
            data_paths.sort()

        if not data_paths:  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        # Just train for every file, whether 4-fold or 8-fold. Too annoying to separate them.
        slice_counts = [self.get_slice_number(str(file_name)) for file_name in data_paths]
        self.num_slices = sum(slice_counts)

        paths_and_slices = list()
        for path, slice_num in zip(data_paths, slice_counts):
            paths_and_slices += [[path, s_idx] for s_idx in range(slice_num)]

        self.paths_and_slices = paths_and_slices
        assert self.num_slices == len(paths_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} initialization!')

    def __len__(self):
        return self.num_slices

    @staticmethod
    def get_slice_number(file_name):  # For data files only, not label files.
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            return hf['data'].shape[0]

    @staticmethod
    def slice_parse_fn(data_path, slice_num):
        assert isinstance(data_path, Path) and isinstance(slice_num, int)

        with h5py.File(data_path, 'r', swmr=True) as dhf:
            ds_slice_arr = np.asarray(dhf['data'][slice_num])
            acc_fac = dhf.attrs['acceleration']
            fat = dhf.attrs['acquisition']
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')

        return ds_slice_arr, acc_fac, fat_supp

    def __getitem__(self, idx):
        file_path, slice_num = self.paths_and_slices[idx]
        ds_slice, acc_fac, fat_supp = self.slice_parse_fn(file_path, slice_num)
        data, mean, std = self.transform(ds_slice, fat_supp)
        return data, mean, std, acc_fac, fat_supp, str(file_path), slice_num


class HDF5Dataset(Dataset):
    def __init__(self, data_dir, transform, batch_size=16, training=True, acc_fac=None, for_eval=False):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.training = training
        self.acc_fac = acc_fac
        self.for_eval = for_eval

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        data_path = Path(self.data_dir)
        file_names = [str(h5) for h5 in data_path.glob('*.h5')]
        file_names.sort()

        if not file_names:  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        print(f'Initializing {data_path.stem}. This might take a minute')
        slice_counts = [self.get_slice_number(file_name) for file_name in file_names]
        self.num_slices = sum(slice_counts)

        names_and_slices = list()

        if self.acc_fac is not None:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, self.acc_fac] for s_idx in range(slice_num)]

        else:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx, choice((4, 8))] for s_idx in range(slice_num)]

        self.names_and_slices = names_and_slices
        assert self.num_slices == len(names_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} initialization!')

    def __len__(self):
        return self.num_slices

    @staticmethod
    def get_slice_number(file_name):
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            try:  # Train and Val
                return hf['1'].shape[0]
            except KeyError:  # Test
                return hf['data'].shape[0]

    @staticmethod
    def h5_slice_parse_fn(file_name, slice_num, acc_fac):
        with h5py.File(file_name, 'r', libver='latest', swmr=True) as hf:
            ds_slice_arr = np.asarray(hf[str(acc_fac)][slice_num])
            gt_slice_arr = np.asarray(hf['1'][slice_num])
            # Fat suppression ('CORPDFS_FBK', 'CORPD_FBK').
            fat = hf.attrs['acquisition']
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')

        return ds_slice_arr, gt_slice_arr, fat_supp

    # There should be 2 modes. 1 mode with just data, labels, processed however necessary.
    # The other mode should have all info necessary for for saving and making a submission file.
    def __getitem__(self, idx):  # Need to add transforms.
        file_name, slice_num, acc_fac = self.names_and_slices[idx]
        ds_slice, gt_slice, fat_supp = self.h5_slice_parse_fn(file_name, slice_num, acc_fac)
        data, labels, mean, std = self.transform(ds_slice, gt_slice, fat_supp)

        if not self.for_eval:
            return data, labels  # Just data and labels here.
        else:  # No need for labels here.
            return data, file_name, slice_num, acc_fac, mean, std


class HDF5TestDataset(Dataset):
    def __init__(self, data_dir, transform, batch_size=16, acc_fac=None):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.acc_fac = acc_fac

        if self.acc_fac is not None:  # Use both if self.acc_fac is None.
            assert self.acc_fac in (4, 8), 'Invalid acceleration factor'

        data_path = Path(self.data_dir)
        file_names = [str(h5) for h5 in data_path.glob('*.h5')]
        file_names.sort()

        if not file_names:  # If the list is empty for any reason
            raise OSError("Sorry! No files present in this directory.")

        print(f'Initializing {data_path.stem}. This might take a minute')
        slice_counts = [self.get_slice_number(file_name) for file_name in file_names]
        self.num_slices = sum(slice_counts)

        names_and_slices = list()

        if self.acc_fac is None:
            for name, slice_num in zip(file_names, slice_counts):
                names_and_slices += [[name, s_idx] for s_idx in range(slice_num)]
        else:
            for name, slice_num in zip(file_names, slice_counts):
                if self.get_acc_fac(name) == self.acc_fac:
                    names_and_slices += [[name, s_idx] for s_idx in range(slice_num)]

        self.names_and_slices = names_and_slices
        assert self.num_slices == len(names_and_slices), 'Error in length'
        print(f'Finished {data_path.stem} initialization!')

    def __len__(self):
        return self.num_slices

    @staticmethod
    def get_slice_number(file_name):
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            try:  # Train and Val
                return hf['1'].shape[0]
            except KeyError:  # Test
                return hf['data'].shape[0]

    @staticmethod
    def h5_test_slice_parse_fn(file_name, slice_num):
        with h5py.File(file_name, 'r', libver='latest', swmr=True) as hf:
            ds_slice = np.asarray(hf['data'][slice_num])
            acc_fac = hf.attrs['acceleration']
            # Fat suppression ('CORPDFS_FBK', 'CORPD_FBK').
            fat = hf.attrs['acquisition']
            if fat == 'CORPDFS_FBK':
                fat_supp = True
            elif fat == 'CORPD_FBK':
                fat_supp = False
            else:
                raise TypeError('Invalid fat suppression/acquisition type!')
        return ds_slice, acc_fac, fat_supp

    @staticmethod
    def get_acc_fac(file_name):
        with h5py.File(name=file_name, mode='r', swmr=True) as hf:
            return hf['acceleration']

    def __getitem__(self, idx):  # Need to add transforms.
        file_name, slice_num = self.names_and_slices[idx]
        ds_slice, acc_fac, fat_supp = self.h5_test_slice_parse_fn(file_name, slice_num)

