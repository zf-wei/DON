#%%
import torch
import numpy as np
import sys
import fipy as fp
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

###############################################################################
import argparse
parser = argparse.ArgumentParser(description="DeepONet with configurable parameters.")
parser.add_argument('--var', type=int, default=0, help='Variant of DeepONet')
parser.add_argument('--struct', type=int, default=2, help='Structure of DeepONet')
parser.add_argument('--boundary_parameter', type=float, default=0, help='Weight parameter for boundary conditions')
# 解析命令行参数
args = parser.parse_args()
var = args.var
struct = args.struct
boundary_parameter = args.boundary_parameter


time_limit = 1
time_step = 0.01
n_points = 50
total_time_steps = int(time_limit/time_step)
total_sample = 500
boundary = int(total_sample*4/5) # 设置训练集和测试集的边界
batch_size = 20
epochs = 3000

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
        1: (3, 3, 100, 50),
        2: (3, 3, 200, 50),
    }
    if struct in structure_params:
        branch_depth, trunk_depth, hidden_dim, output_dim = structure_params[struct]
    else:
        raise ValueError("Invalid structure type")

#%%
# In this cell, we define the function to get the cell centers of a 1D mesh. 
# Also, we set up the spatial and temporal grid points for the training and testing datasets.
# This is the so-called y_expanded tensor. 
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

# Example usage:
cell_centers = get_cell_centers(n_points=n_points)
cell_centers = np.around(cell_centers, decimals=2)

time_steps = np.arange(time_step, time_limit+time_step, time_step)
time_steps = np.around(time_steps, decimals=2)

Y1, Y2 = np.meshgrid(cell_centers, time_steps)  # 第一个变量进行行展开，第二个变量进行列展开

y = np.column_stack([Y2.ravel(),Y1.ravel()]) 
# 先将 Y2 和 Y1 进行展开，然后将展开后的两个向量进行列合并

y_tensor = torch.tensor(y, dtype=torch.float)
print(f"The dimension of y_tensor is {y_tensor.shape}.")
y_expanded = y_tensor.unsqueeze(0).expand(total_sample, -1, -1)
print(f"The dimension of y_expanded is {y_expanded.shape} after expanding.")


#%%
# In this cell, we load the initial conditions and solutions from the saved files.

# Define the directory where you want to save the file
from pathlib import Path
# Get the current directory
current_dir = Path.cwd()
data_directory = os.path.join(current_dir.parent, 'data')

initials_name = f'heat_initials_{len(cell_centers)}.npy'
solutions_name = f'heat_solutions_{len(cell_centers)}.npy'

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
u_expanded = u_expanded.expand(-1, total_time_steps*n_points, -1) # u_expanded: tensor[total_sample, total_time_steps*n_points, n_points]
print(f"The dimension of u_expanded is {u_expanded.shape} after expanding.")

#%%
# I have a tensor of shape (total_sample, n_points) representing the initial conditions.
# In this cell, I wanted to expand it to (total_sample, total_time_steps*n_points) by repeating the
# initial conditions for each time step.

# Assuming u_tensor is the tensor of shape (total_sample, n_points)
# Expand the tensor to (total_sample, total_time_steps*n_points)
u_corresponding = u_tensor.repeat(1, total_time_steps)
u_corresponding = u_corresponding.unsqueeze(2)
# print(u_corresponding.shape)

if var==2 or var==3:
    y_expanded = torch.cat((y_expanded, u_corresponding), dim=-1)

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
train_set = CustomDataset(u_expanded[:boundary], y_expanded[:boundary], s_expanded[:boundary])
test_set = CustomDataset(u_expanded[boundary:], y_expanded[boundary:], s_expanded[boundary:])

# 创建 DataLoader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)


#%%
"""
Design DeepONet Components.
"""
# Branch Network
class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(BranchNet, self).__init__()
        layers = []
        in_dim = input_dim
        
        # 添加多个隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.fc(x)

# Trunk Network
class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(TrunkNet, self).__init__()
        layers = []
        in_dim = input_dim
        
        # 添加多个隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, y):
        return self.fc(y)

#%%
# In this cell, we define the DeepONet Variants
class DeepONet_0(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet_0, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dims, output_dim)

    def forward(self, x, yy):
        branch_output = self.branch_net(x)
        trunk_output = self.trunk_net(yy)
        # Combine the outputs (typically element-wise product)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output


class DeepONet_1(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet_1, self).__init__()
        self.branch_net = BranchNet(branch_input_dim + 1, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dims, output_dim)

    def forward(self, x, yy):
        y_part = yy[:, :, -1].unsqueeze(-1)
        x_extend = torch.cat((x, y_part), dim=-1)
        branch_output = self.branch_net(x_extend)

        trunk_output = self.trunk_net(yy)

        # Combine the outputs (typically element-wise product)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output


class DeepONet_2(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet_2, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim + 1, hidden_dims, output_dim)

    def forward(self, x, yy):
        branch_output = self.branch_net(x)
        trunk_output = self.trunk_net(yy)

        # Combine the outputs (typically element-wise product)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output


class DeepONet_3(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet_3, self).__init__()
        self.branch_net = BranchNet(branch_input_dim + 1, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim + 1, hidden_dims, output_dim)

    def forward(self, x, yy):
        y_part = yy[:, :, -1].unsqueeze(-1)
        x_extend = torch.cat((x, y_part), dim=-1)
        branch_output = self.branch_net(x_extend)

        trunk_output = self.trunk_net(yy)

        # Combine the outputs (typically element-wise product)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output


class DeepONet_4(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet_4, self).__init__()
        self.branch_net = BranchNet(branch_input_dim + 1, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim + branch_input_dim, hidden_dims, output_dim)

    def forward(self, x, yy):
        y_part = yy[:, :, -1].unsqueeze(-1)
        x_extend = torch.cat((x, y_part), dim=-1)
        branch_output = self.branch_net(x_extend)

        yy_extend = torch.cat((yy, x), dim=-1)
        trunk_output = self.trunk_net(yy_extend)

        # Combine the outputs (typically element-wise product)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output

#%%
# In this cell, we define the DeepONet_6, the modified DeepONet
# Branch Network
class MBranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(MBranchNet, self).__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 动态生成多层隐藏层
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Apply weight initialization if needed
        # self.apply(initialize_weights)

    def forward(self, x, U, V):
        xx = self.elu(self.fc1(x))
        xx = torch.mul(1 - xx, U) + torch.mul(xx, V)

        # 动态遍历隐藏层
        for hidden_layer in self.hidden_layers:
            xx = self.elu(hidden_layer(xx))
            xx = torch.mul(1 - xx, U) + torch.mul(xx, V)

        return self.fc_out(xx)


# Trunk Network
class MTrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(MTrunkNet, self).__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 动态生成多层隐藏层
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Apply weight initialization if needed
        # self.apply(initialize_weights)

    def forward(self, y, U, V):
        yy = self.elu(self.fc1(y))
        yy = torch.mul(1 - yy, U) + torch.mul(yy, V)

        # 动态遍历隐藏层
        for hidden_layer in self.hidden_layers:
            yy = self.elu(hidden_layer(yy))
            yy = torch.mul(1 - yy, U) + torch.mul(yy, V)

        return self.fc_out(yy)


# DeepONet_6
class DeepONet_6(nn.Module):
    def __init__(self, branch_input_dim, branch_depth, trunk_input_dim, trunk_depth, hidden_dim, output_dim):
        super(DeepONet_6, self).__init__()
        self.elu = nn.ELU()
        self.branch_net = MBranchNet(branch_input_dim, hidden_dim, output_dim, branch_depth)
        self.trunk_net = MTrunkNet(trunk_input_dim, hidden_dim, output_dim, trunk_depth)
        self.fcB = nn.Linear(branch_input_dim, hidden_dim)
        self.fcT = nn.Linear(trunk_input_dim, hidden_dim)
        # Apply weight initialization
        # self.apply(initialize_weights)

    def forward(self, x, y):
        U = self.elu(self.fcB(x))
        V = self.elu(self.fcT(y))

        branch_output = self.branch_net(x, U, V)
        trunk_output = self.trunk_net(y, U, V)
        # Combine the outputs (typically element-wise product)

        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output

#%%
# In this cell, we define the dictionary mapping the variant number to the corresponding DeepONet model
DeepONets = {
    0: DeepONet_0,
    1: DeepONet_1,
    2: DeepONet_2,
    3: DeepONet_3,
    4: DeepONet_4,
    6: DeepONet_6
}
#%%
# In this cell, we define the loss function

# mean squared error function
def mse(prediction, target):
    ms_loss = torch.mean((prediction - target) ** 2)
    return ms_loss
# boundary condition
def boundary_error(prediction):
    prediction.reshape(-1, total_time_steps, n_points)
    left_boundary = prediction[:, :, 0]
    right_boundary = prediction[:, :, -1]
    both_boundary = torch.cat([left_boundary, right_boundary], dim=1)
    both_boundary_squared = both_boundary ** 2
    return both_boundary_squared.mean()
# loss function
def loss_fn(prediction, target, boundary_parameter=0):
    ms_loss = mse(prediction, target)
    bc_loss = boundary_error(prediction)
    return ms_loss + boundary_parameter * bc_loss
#%%
# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if var!=6:
    model = DeepONets[var](branch_input_dim, trunk_input_dim, hidden_dims, output_dim).to(device)
elif var==6:
    model = DeepONets[var](branch_input_dim, branch_depth, trunk_input_dim, trunk_depth, hidden_dim, output_dim).to(device)

optimizer = optim.Adamax(model.parameters(), lr=0.001)
#%%
# 训练模型
error_list = []
err_best = float('inf')
err_prev = 0
best_epoch = 0
model_best = model.state_dict().copy()


for epoch in range(epochs):
    print(f"Epoch {epoch+1}") 
    err = []
    for input1_batch, input2_batch, target_batch in train_loader:
        input1_batch = input1_batch.to(device)
        input2_batch = input2_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        outputs = model(input1_batch, input2_batch)
        loss = loss_fn(outputs, target_batch, boundary_parameter)
        err.append(loss.item())
        if loss.item()<err_best:
            err_best = loss.item()
            best_epoch = epoch
            model_best = model.state_dict().copy()
            model_filename_best = f"Var{var}_Struct{struct}_Sensor{n_points}_Batch{batch_size}-best.pth"
            torch.save(model_best, model_filename_best)
            print(f"A best model at epoch {epoch+1} has been saved with training error {err_best:.9f}.", file=sys.stderr)
        loss.backward()
        optimizer.step()
        del input1_batch, input2_batch, outputs, loss
        torch.cuda.empty_cache()  # 释放当前批次的缓存
    error_list.append(err)
    err_curr = np.mean(err)
    print(f"Epoch {epoch+1}, Loss: {err_curr:.9f}, Improvement: {err_curr - err_prev:.9f}, Best Loss: {err_best:.9f} in Epoch {best_epoch+1}")
    err_prev = err_curr
    if epoch%50==49:
        # 保存损失值和模型，修改文件名以包含参数信息  
        output_filename = f"Var{var}_Struct{struct}_Sensor{n_points}_Batch{batch_size}-final.npy"
        model_filename = f"Var{var}_Struct{struct}_Sensor{n_points}_Batch{batch_size}-final.pth"
        np.save(output_filename, np.array(error_list))
        torch.save(model.state_dict(), model_filename)
        print(f"Model saving checkpoint: the model trained after epoch {epoch+1} has been saved with the training errors.", file=sys.stderr)
#%%
'''
errs = np.array(error_list)
print(np.mean(errs,axis=1))
'''
#%%
# 保存损失值和模型，修改文件名以包含参数信息
output_filename = f"Var{var}_Struct{struct}_Sensor{n_points}_Batch{batch_size}-final.npy"
model_filename = f"Var{var}_Struct{struct}_Sensor{n_points}_Batch{batch_size}-final.pth"

np.save(output_filename, np.array(error_list))
torch.save(model.state_dict(), model_filename)