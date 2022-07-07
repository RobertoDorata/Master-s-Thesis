import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, hidden_layer_dims, output_layer_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.output_layer_dims = output_layer_dims
        self.n_actions = n_actions
        self.input_layer = nn.Linear(*self.input_dims, self.hidden_layer_dims)
        self.hidden_layer = nn.Linear(self.hidden_layer_dims, self.output_layer_dims)
        self.output_layer = nn.Linear(self.output_layer_dims, self.n_actions)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        actions = self.output_layer(x)

        return actions
