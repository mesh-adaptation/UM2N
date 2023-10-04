# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

__all__ = ['GCN', 'GNN']


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(9, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv4 = GCNConv(16, 16)
        self.conv5 = GCNConv(16, 16)
        self.fc1 = Linear(16 * 5, 24)
        self.fc2 = Linear(24, 8)
        self.fc3 = Linear(8, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x1 = torch.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = torch.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = torch.relu(x3)
        x4 = self.conv4(x3, edge_index)
        x4 = torch.relu(x4)
        x5 = self.conv5(x4, edge_index)
        x5 = torch.relu(x5)
        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x6 = self.fc1(x6)
        x6 = torch.relu(x6)
        x6 = self.fc2(x6)
        x6 = torch.relu(x6)
        x6 = self.fc3(x6)
        x6 = torch.sigmoid(x6)

        return x6


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        self.conv6 = GCNConv(32, 32)
        self.conv7 = GCNConv(32, 16)
        self.conv8 = GCNConv(16, num_classes)
    
    def fix_boundary(self):
        pass

    def forward(self, data):
        x_orig = torch.tensor(data.x, dtype=torch.float)[:, :2]
        x, edge_index = data.x, data.edge_index
        # x = x[:, :3]

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv7(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv8(x, edge_index)
        # x = F.sigmoid(x)

        # add movement x to original x
        x = x + x_orig

        res = x

        return res
