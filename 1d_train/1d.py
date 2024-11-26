#%%
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


import torch.optim as optim
import sys


sys.path.append(os.path.abspath(os.path.expanduser("~/DON")))


import argparse
parser = argparse.ArgumentParser(description="DeepONet with configurable parameters.")
parser.add_argument('--problem', type=str, default="heat", help='Problem to solve')
parser.add_argument('--var', type=int, default=0, help='Variant of DeepONet')
parser.add_argument('--visc', type=float, default=0.0001, help='Viscosity')
parser.add_argument('--struct', type=int, default=1, help='Structure of DeepONet')
parser.add_argument('--sensor', type=int, default=50, help='Number of sensors')
parser.add_argument('--boundary_parameter', type=float, default=0, help='Weight parameter for boundary conditions')
parser.add_argument('--initial_parameter', type=float, default=0, help='Weight parameter for initial conditions')
parser.add_argument('--train_batch_size', type=int, default=8000, help='Train Batch size')
parser.add_argument('--test_batch_size', type=int, default=2000, help='Train Batch size')
# 解析命令行参数
args = parser.parse_args()
problem = args.problem
var = args.var
visc = args.visc
struct = args.struct
n_points = args.sensor
boundary_parameter = args.boundary_parameter
initial_parameter = args.initial_parameter
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size


epochs = 5
## 需要修改
#%%
# In this cell, we define the configurable parameters for the DeepONet

time_limit = 1
time_step = 0.01

if problem=="heat":
    time_start = time_step
    total_time_steps = int(time_limit/time_step)
    from utilities.tools import get_cell_centers
    evaluating_points = get_cell_centers(n_points=n_points)
elif problem=="burgers":
    time_start = 0
    total_time_steps = (int(time_limit/time_step)+1)
    evaluating_points = np.linspace(0, 1, n_points)

evaluating_points = np.around(evaluating_points, decimals=2)

total_sample = 500
border = int(total_sample * 4 / 5) # 设置训练集和测试集的边界


# Hyperparameters
branch_input_dim = n_points  # Number of points to represent the original function
trunk_input_dim = 2     # Coordinate where we evaluate the transformed function

# Define the dictionary mapping struct values to neural network structures
if var!=6:
    structures = {
        1: {'hidden_dims': [100, 100, 100, 100], 'output_dim': 50},
        2: {'hidden_dims': [200, 200, 200, 200], 'output_dim': 50}
    }

    # Get the configuration based on the struct value
    config = structures.get(struct, {'hidden_dims': [], 'output_dim': 0})

    hidden_dims = config['hidden_dims']
    output_dim = config['output_dim']
elif var==6:
    structure_params = {
        1: (4, 4, 100, 50),
        2: (4, 4, 200, 50),
    }
    if struct in structure_params:
        branch_depth, trunk_depth, hidden_dim, output_dim = structure_params[struct]
    else:
        raise ValueError("Invalid structure type")
#%%
# In this cell, we import the function to get the cell centers of a 1D mesh.
# Also, we set up the spatial and temporal grid points for the training and testing datasets.
# This is the so-called y_expanded tensor.
time_steps = np.arange(time_start, time_limit+time_step, time_step)
time_steps = np.around(time_steps, decimals=2)

Y1, Y2 = np.meshgrid(evaluating_points, time_steps)  # 第一个变量进行行展开，第二个变量进行列展开

y = np.column_stack([Y2.ravel(),Y1.ravel()]) 
# 先将 Y2 和 Y1 进行展开，然后将展开后的两个向量进行列合并

y_tensor = torch.tensor(y, dtype=torch.float)
print(f"The dimension of y_tensor is {y_tensor.shape}.")
y_expanded = y_tensor.unsqueeze(0).expand(total_sample, -1, -1)
print(f"The dimension of y_expanded is {y_expanded.shape} after expanding.")
print("The zero coordinate of y_expanded is time and the first coordinate is space.")
#%%
# In this cell, we load the initial conditions and solutions from the saved files.

# Define the directory where you want to save the file
from pathlib import Path
# Get the current directory
current_dir = Path.cwd()
data_directory = os.path.join(current_dir.parent, 'data')
## 需要修改
# data_directory = os.path.join(current_dir, 'data')
initials_name = f'{problem}_initials_{visc}_{len(evaluating_points)}.npy'
solutions_name = f'{problem}_solutions_{visc}_{len(evaluating_points)}.npy'

# Define the file paths
initials_path = os.path.join(data_directory, initials_name)
solutions_path = os.path.join(data_directory, solutions_name)

# Load the data
initials = np.load(initials_path)
solutions = np.load(solutions_path)

print(f"The dimensions of the initial conditions are: {initials.shape}")
print(f"The dimensions of the solutions are: {solutions.shape}")
#%%
# In this cell, we arrange the initial conditions into the desired format for training the DeepONet.
# This is the so-called u_expanded tensor.
u_tensor = torch.tensor(initials, dtype=torch.float)
print(f"The dimension of u_tensor is {u_tensor.shape}.")

u_expanded = u_tensor.unsqueeze(1) # u_expanded: tensor[total_sample, 1, n_points]
u_expanded = u_expanded.expand(-1, total_time_steps * n_points, -1) # u_expanded: tensor[total_sample, total_time_steps*n_points, n_points]
print(f"The dimension of u_expanded is {u_expanded.shape} after expanding.")
#%%
# I have a tensor of shape (total_sample, n_points) representing the initial conditions. In this cell, I wanted to expand it to (total_sample, total_time_steps*n_points) by repeating the initial conditions for each time step.

# Assuming u_tensor is the tensor of shape (total_sample, n_points)
# Expand the tensor to (total_sample, total_time_steps*n_points)
u_corresponding = u_tensor.repeat(1, total_time_steps)
u_corresponding = u_corresponding.unsqueeze(2) # This is the so-called corresponding initial value

# Take the spatial coordinate of the y_expanded tensor
y_space = y_expanded[:, :, 1].unsqueeze(-1)
#%%
# In this cell, we modify the input of the DeepONet based on the variant chosen.
# We also update the input dimensions of the DeepONet based on the variant chosen.
if var==2 or var==3:
    y_expanded = torch.cat((y_expanded, u_corresponding), dim=-1)
elif var==4:
    y_expanded = torch.cat((y_expanded, u_expanded), dim=-1)

if var== 1 or var==3 or var==4:
    u_expanded = torch.cat((u_expanded, y_space), dim=-1)

var_mapping = {
    1: {'var_branch_input_dim': branch_input_dim + 1, 'var_trunk_input_dim': trunk_input_dim},
    2: {'var_branch_input_dim': branch_input_dim, 'var_trunk_input_dim': trunk_input_dim + 1},
    3: {'var_branch_input_dim': branch_input_dim + 1, 'var_trunk_input_dim': trunk_input_dim + 1},
    4: {'var_branch_input_dim': branch_input_dim + 1, 'var_trunk_input_dim': trunk_input_dim + branch_input_dim}
}

if var in var_mapping:
    branch_input_dim = var_mapping[var]['var_branch_input_dim']
    trunk_input_dim = var_mapping[var]['var_trunk_input_dim']
#%%
# In this cell, we arrange the solutions into the desired format for training the DeepONet.
# This is the so-called s_expanded tensor.

solutions_linear = np.zeros((total_sample, total_time_steps*n_points))

for i in range(total_sample):
    solutions_linear[i] = solutions[i].flatten()

# solutions is a 3D array of shape (total_sample, total_time_steps, n_points)
print(f"The loaded solution dataset has dimension {solutions.shape},\n\t while the arranged linearized dataset has dimension {solutions_linear.shape}.")

s_tensor  = torch.tensor(solutions_linear, dtype=torch.float) # s_tensor: tensor[total_sample, total_time_steps*n_points]
s_expanded  = s_tensor.unsqueeze(2) # s_expanded: tensor[total_sample, total_time_steps*n_points, 1]

print(f"The dimension of s_tensor is {s_tensor.shape}.")
print(f"The dimension of s_expanded is {s_expanded.shape} after expanding.")
#%%
# In this cell, we create the training and testing datasets and dataloader for the DeepONet.

u_train = u_expanded[:border]
y_train = y_expanded[:border]
s_train = s_expanded[:border]

u_test = u_expanded[border:]
y_test = y_expanded[border:]
s_test = s_expanded[border:]

u_train_combined = u_train.reshape(-1, u_train.shape[-1])
y_train_combined = y_train.reshape(-1, y_train.shape[-1])
s_train_combined = s_train.reshape(-1, s_train.shape[-1])

u_test_combined = u_test.reshape(-1, u_test.shape[-1])
y_test_combined = y_test.reshape(-1, y_test.shape[-1])
s_test_combined = s_test.reshape(-1, s_test.shape[-1])

from utilities.tools import CustomDataset as CustomDataset

train_set = CustomDataset(u_train_combined, y_train_combined, s_train_combined)
test_set = CustomDataset(u_test_combined, y_test_combined, s_test_combined)

# 创建 DataLoader
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=1)

print(f"The training dataset has {len(train_set)} samples, while the train_loader has {len(train_loader)} batches.")
print(f"The training dataset has {len(test_set)} samples, while the train_loader has {len(test_loader)} batches.")
#%%
# In this cell, we import and set up the model and load the trained parameters
# The loss function is also imported here.

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if var!=6:
    from utilities.DeepONets import DeepONet
    model = DeepONet(branch_input_dim, trunk_input_dim, hidden_dims, output_dim).to(device)
elif var==6:
    from utilities.DeepONets import ModifiedDeepONet
    model = ModifiedDeepONet(branch_input_dim, branch_depth, trunk_input_dim, trunk_depth, hidden_dim, output_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

from utilities.loss_fns import loss_fn_1d_combined as loss_fn
#%%
# 训练模型
error_list = []
time_list = []
err_best = float('inf')
err_prev = 0
best_epoch = 0
model_best = model.state_dict().copy()

import time

for epoch in range(epochs):
    start_time = time.time()  # Record the start time
    print(f"Epoch {epoch+1}") 
    err = []
    for input1_batch, input2_batch, target_batch in train_loader:
        input1_batch = input1_batch.to(device)
        input2_batch = input2_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        outputs = model(input1_batch, input2_batch)
        loss = loss_fn(outputs, target_batch, boundary_parameter, initial_parameter, total_time_steps, n_points)
        err.append(loss.item())
        if loss.item()<err_best:
            err_best = loss.item()
            best_epoch = epoch
            model_best = model.state_dict().copy()
            model_params_best = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-best.pth"
            torch.save(model_best, model_params_best)
            print(f"A best model at epoch {epoch+1} has been saved with training error {err_best:.14f}.", file=sys.stderr)
        loss.backward()
        optimizer.step()
        del input1_batch, input2_batch, outputs, loss
        torch.cuda.empty_cache()  # 释放当前批次的缓存
    error_list.append(err)
    err_curr = np.mean(err)
    epoch_time = time.time() - start_time  # Calculate the elapsed time
    time_list.append(epoch_time)

    print(f"Epoch {epoch+1}, Loss: {err_curr:.14f}, Improvement: {err_curr - err_prev:.14f}, Best Loss: {err_best:.14f} in Epoch {best_epoch+1}, Time: {epoch_time:.2f} seconds")

    err_prev = err_curr
    if epoch%50==49:
        # 保存损失值和模型
        training_error_list = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-final.npy"
        model_params_final = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-final.pth"
        np.save(training_error_list, np.array(error_list))
        torch.save(model.state_dict(), model_params_final)
        print(f"Model saving checkpoint: the model trained after epoch {epoch+1} has been saved with the training errors.", file=sys.stderr)

    sys.stdout.flush()
#%%
# errs = np.array(error_list)

# print(np.mean(errs,axis=1))

## 需要修改
#%%
# 保存损失值和模型
training_error_list = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-final.npy"
training_time_list = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-final.time"
model_params_final = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-final.pth"

np.save(training_error_list, np.array(error_list))
np.save(training_time_list, np.array(time_list))
torch.save(model.state_dict(), model_params_final)
#%%
# # In this cell, we compute training and testing loss for the model
def compute_loss(model, data_loader, device, description="Computing loss"):
    """
    Compute the loss for a given dataset using the trained model.

    Parameters:
    - model: The trained model.
    - data_loader: DataLoader for the dataset (train or test).
    - device: Device to run the computations on (CPU or GPU).
    - description: Description for the tqdm progress bar.

    Returns:
    - average_loss: The average loss over the dataset.
    """
    total_loss = 0
    with torch.no_grad():
        for input1_batch, input2_batch, target_batch in data_loader:
            input1_batch = input1_batch.to(device)
            input2_batch = input2_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model(input1_batch, input2_batch)
            loss = loss_fn(outputs, target_batch, boundary_parameter, initial_parameter, total_time_steps, n_points)
            total_loss += loss.item()
            del input1_batch, input2_batch, target_batch, outputs
            torch.cuda.empty_cache()  # Release cache for the current batch
    average_loss = total_loss / len(data_loader)
    return average_loss
#%%

training_loss_best = f"{problem}_Var{var}_Visc{visc}_Struct{struct}_Sensor{n_points}_Boundary{boundary_parameter}_Initial{initial_parameter}_Batch{train_batch_size}-best.loss"

model.load_state_dict(torch.load(model_params_best, map_location=torch.device(device), weights_only=True))
model.eval()

training_loss = compute_loss(model, train_loader, device, description="Computing training loss")
testing_loss = compute_loss(model, test_loader, device, description="Computing testing loss")

print(f"With the best parameters, the training loss is {training_loss:.14f}, while the testing loss is {testing_loss:.14f}.")

with open(training_loss_best, 'w') as f:
    f.write("With the best parameters, the training and testing losses for the model are as follows:\n")
    f.write(f"Train loss: {training_loss}\n")
    f.write(f"Test loss: {testing_loss}\n")