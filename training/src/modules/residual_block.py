import torch.nn as nn

class ResidualBlock(nn.Sequential):

    def __init__(self, in_dimension: int, out_dimension: int):
        super().__init__()
        self.add_module("linear", nn.Linear(in_dimension, out_dimension))
        self.add_module("bn", nn.BatchNorm1d(out_dimension))
        self.add_module("relu", nn.LeakyReLU())

    def forward(self, x):
        return x + super().forward(x)