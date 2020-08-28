import torch
from torch import nn
import os

############################################
############################################

class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, size, activation=torch.tanh, output_activation=None):
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        in_ = input_size
        for i in range(n_layers):
            self.layers.append(nn.Linear(in_, size))
            in_ = size

        self.layers.append(nn.Linear(size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))

        return self.output_activation(x)


############################################
############################################


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
