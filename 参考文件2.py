import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description="DeepONet with configurable parameters.")
parser.add_argument('--stretch_para', type=int, default=50, help='Stretch parameter')
parser.add_argument('--loss_weight', type=float, default=10.0, help='Weight for edge loss')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for DataLoader')
parser.add_argument('--data', type=int, choices=[30, 81], required=True, help='Data points per function (30 or 81)')
parser.add_argument('--strc', type=int, choices=[1, 2], required=True, help='Structure type (1 or 2)')
parser.add_argument('--crit', type=int, choices=[1, 2], required=True, help='Loss criterion (1 for MMSE, 2 for relative error)')

# 解析命令行参数
args = parser.parse_args()

# 提取 loss_crit 参数
crit = args.crit

# 提取参数
stretch_para = args.stretch_para
loss_weight = args.loss_weight
batch_size = args.batch_size
data = args.data
strc = args.strc

# 设置数据集相关参数
if data == 30:
    data_id = 29
    n_points = 30  # 每个函数的采样点数量
    total_sample = 1000
    epochs = 5000
elif data == 81:
    data_id = 26
    n_points = 81  # 每个函数的采样点数量
    total_sample = 500
    epochs = 1000

# 设置网络结构
structure_params = {
    1: ([100, 100, 100], 50),
    2: ([200, 200, 200, 200], 100),
    3: ([100, 100, 100, 100], 50)
}
if strc in structure_params:
    hidden_dims, output_dim = structure_params[strc]
else:
    raise ValueError("Invalid structure type")


# 加载数据
home_dir = os.path.expanduser("~")  # 获取 home 目录
source_path = os.path.join(home_dir, f"DLL/data/202409{data_id}_source_data.npz")  # 得到 source 数据集的路径
solution_path = os.path.join(home_dir, f"DLL/data/202409{data_id}_solution_data.npz")  # 得到 solution 数据集的路径

source_data = np.load(source_path)
sources = stretch_para * source_data['sources']
solution_data = np.load(solution_path)
solutions = stretch_para * solution_data['solutions']

boundary = int(total_sample * 9 / 10)  # 划分训练集和测试集的边界

# 构建线性化的 1D 数据
sources_linear = np.zeros((total_sample, n_points * n_points))
solutions_linear = np.zeros((total_sample, n_points * n_points))
for i in range(total_sample):
    sources_linear[i] = sources[i].flatten()
    solutions_linear[i] = solutions[i].flatten()

# 处理 y 数据，构建 2D 网格
y1 = np.linspace(0, 1, n_points)
y2 = np.linspace(0, 1, n_points)
Y1, Y2 = np.meshgrid(y1, y2)
y = np.column_stack([Y1.ravel(), Y2.ravel()])
y_tensor = torch.tensor(y, dtype=torch.float)  # 转为 tensor
y_expanded = y_tensor.unsqueeze(0).expand(total_sample, -1, -1)  # 扩展形状

# 准备输入数据
u_tensor = torch.tensor(sources_linear[:total_sample], dtype=torch.float)  # 随机函数
u_expanded = u_tensor.unsqueeze(1).expand(-1, y_tensor.size(0), -1)  # 保持鸳鸯扩展形状
s_tensor = torch.tensor(solutions_linear[:total_sample], dtype=torch.float)  # 解函数
s_expanded = s_tensor.unsqueeze(2)  # 扩展形状确保保持相同维度

# 构建自定义数据集
class CustomDataset(Dataset):
    def __init__(self, input1_data, input2_data, targets):
        self.input1_data = input1_data
        self.input2_data = input2_data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.input1_data[idx], self.input2_data[idx], self.targets[idx]

# 创建训练和测试集
train_set = CustomDataset(u_expanded[:boundary], y_expanded[:boundary], s_expanded[:boundary])
test_set = CustomDataset(u_expanded[boundary:], y_expanded[boundary:], s_expanded[boundary:])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# 构建 DeepONet 网络
class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(BranchNet, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.fc(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(TrunkNet, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, y):
        return self.fc(y)

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dims, output_dim)
        
    def forward(self, x, y):
        branch_output = self.branch_net(x)
        trunk_output = self.trunk_net(y)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 鸳鸯式内积操作
        return output

# 定义损失函数
def edge_mean_square(tensor):
    top = tensor[:, 0, :]
    bottom = tensor[:, -1, :]
    left = tensor[:, :, 0]
    right = tensor[:, :, -1]
    edges = torch.cat([top, bottom, left, right], dim=1)
    edges_squared = edges ** 2
    return edges_squared.mean().item()

def mmse_absolute(prediction, target):
    mse = torch.mean((prediction - target) ** 2)
    prediction = prediction.reshape(-1, n_points, n_points)
    err_bd = edge_mean_square(prediction)
    return mse + loss_weight * err_bd

def mmse_relative(prediction, target, epsilon=1e-6):
    # 计算均方误差
    mse = torch.sum((prediction - target) ** 2) 
    # 计算边界误差
    prediction = prediction.reshape(-1, n_points, n_points)
    err_bd = edge_mean_square(prediction)
    # 在相对误差的分母中加入 epsilon 以处理小值问题
    loss = mse / (torch.sum(target ** 2)) + loss_weight * err_bd
    return loss


if crit == 1:
    mmse=mmse_absolute
elif crit == 2:
    mmse=mmse_relative

# 设置超参数和模型
branch_input_dim = n_points * n_points
trunk_input_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepONet(branch_input_dim, trunk_input_dim, hidden_dims, output_dim).to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

# 训练模型
error_list = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    err = []
    for input1_batch, input2_batch, target_batch in train_loader:
        input1_batch = input1_batch.to(device)
        input2_batch = input2_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()
        outputs = model(input1_batch, input2_batch)
        loss = mmse(outputs, target_batch)
        err.append(loss.item())

        loss.backward()
        optimizer.step()
        del input1_batch, input2_batch, target_batch, outputs, loss
        torch.cuda.empty_cache()  # 释放当前批次的缓存
    error_list.append(err)

# 保存损失值和模型，修改文件名以包含参数信息
output_filename = f"mod0_crit{crit}_data{data}_strc{strc}_stretch{stretch_para}_batch{batch_size}_loss_weight{loss_weight}.npy"
model_filename = f"mod0_crit{crit}_data{data}_strc{strc}_stretch{stretch_para}_batch{batch_size}_loss_weight{loss_weight}.pth"

np.save(output_filename, np.array(error_list))
torch.save(model.state_dict(), model_filename)
