import torch
from torch import nn


class TripleLayerNet(nn.Module):
    def __init__(self, N_features):
        super(TripleLayerNet, self).__init__()
        self.l1 = nn.Linear(N_features, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, x, mask, indices):
        # x = torch.sigmoid(self.l1(x))
        x = self.l1(x)
        x = torch.sigmoid(self.l2(x))
        x = self.l3(x)
        return x
