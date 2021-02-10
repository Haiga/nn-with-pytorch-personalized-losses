import torch
from torch import nn
import torch.nn.functional as F


# class DoubleLayerNet(nn.Module):
#     def __init__(self, N_features):
#         super(DoubleLayerNet, self).__init__()
#         self.l1 = nn.Linear(N_features, N_features)
#         self.l2 = nn.Linear(N_features, 1)
#
#     def forward(self, x, mask, indices):
#         x = torch.sigmoid(self.l1(x))
#         x = self.l2(x)
#         return x


# class DoubleLayerNet(nn.Module):
#
#     def __init__(self, input_size):
#         super(DoubleLayerNet, self).__init__()
#         self.l1 = nn.Linear(input_size, 2*input_size)
#         self.relu = nn.ReLU()
#         self.l3 = nn.Linear(2*input_size, input_size)
#         # self.relu1 = nn.Dropout(p=0.1)
#         self.relu1 = nn.ReLU()
#         self.l4 = nn.Linear(input_size, 1)
#
#     def forward(self, x, mask, indices):
#         x = self.l1(x)
#         x = self.relu(x)
#         x = self.l3(x)
#         x = self.relu1(x)
#         x = self.l4(x)
#         return x


# class DoubleLayerNet(nn.Module):
#     def __init__(self, input_size):
#         super(DoubleLayerNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         # self.fc2 = nn.Linear(64, 64)
#         # self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, 1)
#
#     def forward(self, x, c1, c2):
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return F.log_softmax(x, dim=1)


class DoubleLayerNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, c1, c2):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def predict(self, x, c1, c2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x), dim=1)
        x = self.fc3(x)
        return x
