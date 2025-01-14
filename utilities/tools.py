# Description: This file contains utility functions that are used in the whole project.

#%%
import numpy as np
import fipy as fp

def get_cell_centers(time_limit = 1, n_points = 50):
    """
    Get the cell center positions for a 1D mesh with the specified number of grid points.

    Parameters:
    - n_points: Number of grid points in the spatial domain.

    Returns:
    - cell_centers: The x-positions of the cell centers.
    """
    L = time_limit  # Length of the domain
    dx = L / n_points

    # Create a 1D mesh
    mesh = fp.Grid1D(nx=n_points, dx=dx)

    # Get the cell center positions
    cell_centers = mesh.cellCenters[0]  # These are the x-positions of the cell centers
    cell_centers = np.array(cell_centers)

    return cell_centers

#%%
from torch.utils.data import Dataset
"""
This is the function to well organize the dataset
"""
class CustomDataset(Dataset):
    def __init__(self, input1_data, input2_data, targets):
        self.input1_data = input1_data
        self.input2_data = input2_data
        self.targets = targets

    def __len__(self):
        return len(self.input1_data)

    def __getitem__(self, idx):
        input1 = self.input1_data[idx]
        input2 = self.input2_data[idx]
        target = self.targets[idx]
        return input1, input2, target

#%%
class DataGenerator(Dataset):
    def __init__(self, u, y, s, batch_size=64, seed=2025):
        """
        Initialization
        :param u: Input feature tensor (N, d1)
        :param y: Input label tensor (N, d2)
        :param s: Output tensor (N, d3)
        :param batch_size: Number of samples per batch
        :param seed: Seed for random number generator
        """
        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]  # Total number of samples
        self.batch_size = batch_size
        self.key = np.random.default_rng(seed)  # Random number generator with a seed

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self.N // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: Not used directly but required by PyTorch Dataset
        """
        # Generate random indices for the batch
        indices = self.key.choice(self.N, self.batch_size, replace=False)
        u_batch, y_batch, s_batch = self.__data_generation(indices)
        return u_batch, y_batch, s_batch

    def __data_generation(self, indices):
        """
        Generates data containing batch_size samples
        :param indices: Random indices for batch sampling
        """
        s_batch = self.s[indices, :]  # Target variable batch
        y_batch = self.y[indices, :]  # Input label batch
        u_batch = self.u[indices, :]  # Input feature batch

        return u_batch, y_batch, s_batch




