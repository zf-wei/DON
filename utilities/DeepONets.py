import torch
import torch.nn as nn
import torch.nn.functional as F

# This is the vanilla DeepONet
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim, activation=F.elu):
        super(DeepONet, self).__init__()
        self.activation = activation

        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.branch_net.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.branch_net.append(nn.Linear(hidden_dims[-1], output_dim))

        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.trunk_net.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.trunk_net.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, u, y):
        for k in range(len(self.branch_net)-1):
            u = self.activation(self.branch_net[k](u))
            y = self.activation(self.trunk_net[k](y))

        branch_output = self.branch_net[-1](u)
        trunk_output = self.trunk_net[-1](y)
        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)

        return output

# This is the modified DeepONet
class ModifiedDeepONet(nn.Module):
    def __init__(self, branch_input_dim, branch_depth, trunk_input_dim, trunk_depth, hidden_dim, output_dim, activation=F.elu):
        super(ModifiedDeepONet, self).__init__()
        self.activation = activation

        self.branch_net = nn.ModuleList()
        self.branch_net.append(nn.Linear(branch_input_dim, hidden_dim))
        for i in range(branch_depth-1):
            self.branch_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net.append(nn.Linear(hidden_dim, output_dim))

        self.trunk_net = nn.ModuleList()
        self.trunk_net.append(nn.Linear(trunk_input_dim, hidden_dim))
        for i in range(trunk_depth-1):
            self.trunk_net.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net.append(nn.Linear(hidden_dim, output_dim))

        self.U1 = nn.Linear(branch_input_dim, hidden_dim)
        self.U2 = nn.Linear(trunk_input_dim, hidden_dim)

    def forward(self, u, y):
        U = self.activation(self.U1(u))
        V = self.activation(self.U2(y))

        for k in range(len(self.branch_net)-1):
            B = self.activation(self.branch_net[k](u))
            T = self.activation(self.trunk_net[k](y))

            u = B * U + (1 - B) * V
            y = T * U + (1 - T) * V

        branch_output = self.branch_net[-1](u)
        trunk_output = self.trunk_net[-1](y)

        output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)

        return output

