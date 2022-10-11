import h5py
import numpy as np
from torch.utils.data import Dataset

# @class    : getTrainDataset
# @brief    : It returns data count and data by reading data from h5 file.
# @param    :
#   h5_file : training dataset h5 file

class getTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(getTrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

# @class    : getValDataset
# @brief    : It returns data count and data by reading data from h5 file.
# @param    :
#   h5_file : validation dataset h5 file

class getValDataset(Dataset):
    def __init__(self, h5_file):
        super(getValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


