import torch
from torch import nn


class DoubleLayerNet(nn.Module):
    def __init__(self, N_features):
        super(DoubleLayerNet, self).__init__()
        self.l1 = nn.Linear(N_features, 10)
        self.l2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = self.l2(x)
        return x
