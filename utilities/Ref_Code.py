import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepONet(nn.Module):
    def __init__(self, input_dim_branch, input_dim_trunk, hidden_dim_branch=128, hidden_dim_trunk=128, output_dim=1):
        super(DeepONet, self).__init__()

        # Branch network
        self.branch_net = nn.Sequential(
            nn.Linear(input_dim_branch, hidden_dim_branch),
            nn.ReLU(),
            nn.Linear(hidden_dim_branch, hidden_dim_branch),
            nn.ReLU(),
            nn.Linear(hidden_dim_branch, hidden_dim_branch)
        )

        # Trunk network
        self.trunk_net = nn.Sequential(
            nn.Linear(input_dim_trunk, hidden_dim_trunk),
            nn.ReLU(),
            nn.Linear(hidden_dim_trunk, hidden_dim_trunk),
            nn.ReLU(),
            nn.Linear(hidden_dim_trunk, hidden_dim_branch)  # Matches output of branch
        )

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim_branch, output_dim)

    def forward(self, branch_input, trunk_input):
        # Pass through branch and trunk networks
        branch_out = self.branch_net(branch_input)  # Shape: (batch_size, hidden_dim_branch)
        trunk_out = self.trunk_net(trunk_input)  # Shape: (batch_size, hidden_dim_branch)

        # Compute dot product and pass through final layer
        combined = branch_out * trunk_out  # Element-wise multiplication
        output = self.output_layer(combined)

        return output

# Define the neural net
class MLP(nn.Module):
    def __init__(self, layers, activation=F.relu):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, inputs):
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        outputs = self.layers[-1](inputs)
        return outputs


class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation=F.relu):
        super(ModifiedMLP, self).__init__()
        self.activation = activation
        self.params = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.params.append(nn.Linear(layers[i], layers[i + 1]))

        self.U1 = nn.Linear(layers[0], layers[1])
        self.U2 = nn.Linear(layers[0], layers[1])

    def forward(self, inputs):
        U = self.activation(self.U1(inputs))
        V = self.activation(self.U2(inputs))
        for layer in self.params[:-1]:
            outputs = self.activation(layer(inputs))
            inputs = outputs * U + (1 - outputs) * V
        final_layer = self.params[-1]
        outputs = final_layer(inputs)
        return outputs


class ModifiedDeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers, activation=F.relu):
        super(ModifiedDeepONet, self).__init__()
        self.activation = activation

        # Initialize branch and trunk layers
        self.branch_params = nn.ModuleList()
        for i in range(len(branch_layers) - 1):
            self.branch_params.append(nn.Linear(branch_layers[i], branch_layers[i + 1]))

        self.trunk_params = nn.ModuleList()
        for i in range(len(trunk_layers) - 1):
            self.trunk_params.append(nn.Linear(trunk_layers[i], trunk_layers[i + 1]))

        # Additional components U1 and U2
        self.U1 = nn.Linear(branch_layers[0], branch_layers[1])
        self.U2 = nn.Linear(trunk_layers[0], trunk_layers[1])

    def forward(self, u, y):
        U = self.activation(self.U1(u))
        V = self.activation(self.U2(y))
        for k in range(len(self.branch_params) - 1):
            branch_layer = self.branch_params[k]
            trunk_layer = self.trunk_params[k]

            B = self.activation(branch_layer(u))
            T = self.activation(trunk_layer(y))

            u = B * U + (1 - B) * V
            y = T * U + (1 - T) * V

        # Final layers
        branch_final = self.branch_params[-1]
        trunk_final = self.trunk_params[-1]

        B = branch_final(u)
        T = trunk_final(y)

        outputs = torch.sum(B * T, dim=-1)  # Sum over the last dimension
        return outputs
