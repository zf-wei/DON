#%%
import torch

# In this cell, we define the loss function

# mean squared error function
def mse(prediction, target):
    ms_loss = torch.mean((prediction - target) ** 2)
    return ms_loss
# boundary condition
def boundary_error_1d(prediction, target, total_time_steps=100, n_points=50):
    prediction_reshape = prediction.reshape(-1, total_time_steps, n_points)
    target_reshape = target.reshape(-1, total_time_steps, n_points)
    left_boundary = prediction_reshape[:, :, 0] - target_reshape[:, :, 0]
    right_boundary = prediction_reshape[:, :, -1] - target_reshape[:, :, -1]
    both_boundary = torch.cat([left_boundary, right_boundary], dim=1)
    both_boundary_squared = both_boundary ** 2
    return both_boundary_squared.mean()
# loss function
def loss_fn_1d(prediction, target, boundary_parameter=0, initial_parameter=0, total_time_steps=100, n_points=50):
    ms_loss = mse(prediction, target)
    bc_loss = boundary_error_1d(prediction, target, total_time_steps, n_points)
    ic_loss = mse(prediction[:, 0, :], target[:, 0, :])
    return ms_loss + boundary_parameter * bc_loss + initial_parameter * ic_loss