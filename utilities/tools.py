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
class CustomDataset_data(Dataset):
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

