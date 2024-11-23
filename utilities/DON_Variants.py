#%%
import torch
import torch.nn as nn

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
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dims, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dims, output_dim)

    def forward(self, x, y):
        branch_output = self.branch_net(x)
        trunk_output = self.trunk_net(y)
        # Combine the outputs (typically element-wise product)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)  # 按照最后一个坐标做内积
        return output

#%%
# In this cell, we define the modified DeepONet
# Branch Network
class mBranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(mBranchNet, self).__init__()
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
class mTrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(mTrunkNet, self).__init__()
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
class mDeepONet(nn.Module):
    def __init__(self, branch_input_dim, branch_depth, trunk_input_dim, trunk_depth, hidden_dim, output_dim):
        super(mDeepONet, self).__init__()
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
