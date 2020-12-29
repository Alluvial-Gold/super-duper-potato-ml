

import numpy as np

import torch
import torch.nn as nn

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network
    layers = []
    for layer_number, (layer_size, next_layer_size) in enumerate(zip(sizes, sizes[1:])):
        act = activation if layer_number < len(sizes)-2 else output_activation
        layers += [nn.Linear(layer_size, next_layer_size), act()]
    return nn.Sequential(*layers)

