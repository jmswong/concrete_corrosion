import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Data(Dataset):
    '''
    Data class for corrosion data, concrete properties, and target label
    '''
    def __init__(self, corrosion_data, target_labels):
        '''
        The first 2 columns are the simulation_idx and timestep respectively,
        and should not be used in training. Columns 3, 4, 5, and 6 are
        floating-point representations of certain concrete properties
        (in particular rebar, cover, tensile_strength, w_c, respectively).
        Columns 7+ are the corrosion depths along the rebar.
        '''
        self.corrosion_inputs = torch.from_numpy(corrosion_data[:, 6:].astype(
            np.float32))
        self.concrete_inputs = torch.from_numpy(corrosion_data[:, 2:6].astype(
            np.float32))
        self.target_labels = torch.from_numpy(target_labels.astype(np.float32))
        self.len = self.corrosion_inputs.shape[0]

    def __getitem__(self, index):
        return self.corrosion_inputs[index], self.concrete_inputs[
            index], self.target_labels[index]

    def __len__(self):
        return self.len


def get_data_loader(X, y, batch_size=None):
    '''
    Returns a DataLoader from input numpy arrays.

    Args:
        X (ndarray): Array of dim (n, 343) containing corrosion depth and
            concrete propoerties features.
        y (ndarray): Array of dim (n, 1) containing labels.
        batch_size (int): Batch size. If None, uses the full input X as a
            single batch.

    Returns
        data_loader: Torch DataLoader object.
    '''
    if batch_size is None:
        batch_size = X.shape[0]
    data = Data(X, y)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    return data_loader
