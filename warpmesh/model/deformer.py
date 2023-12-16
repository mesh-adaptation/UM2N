# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os 
import sys
from torch_geometric.nn import GATv2Conv, MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F


cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)

__all__ = ['RecurrentGATConv']


class RecurrentGATConv(MessagePassing):
    """
    Implements a Recurrent Graph Attention Network (GAT) Convolution layer.

    Attributes:
        to_hidden (GATv2Conv): Graph Attention layer.
        to_coord (nn.Sequential): Output layer for coordinates.
        activation (nn.SELU): Activation function.
    """
    def __init__(self, coord_size=2,
                 hidden_size=512,
                 heads=6, concat=False
                 ):
        super(RecurrentGATConv, self).__init__()
        # GAT layer
        self.to_hidden = GATv2Conv(
            in_channels=coord_size+hidden_size,
            out_channels=hidden_size,
            heads=heads,
            concat=concat
        )
        # output coord layer
        self.to_coord = nn.Sequential(
            nn.Linear(hidden_size, 2),
        )
        # activation function
        self.activation = nn.SELU()

    def forward(self, coord, hidden_state, edge_index):
        # find boundary
        self.find_boundary(coord)
        # Recurrent GAT
        in_feat = torch.cat((coord, hidden_state), dim=1)
        hidden = self.to_hidden(in_feat, edge_index)
        hidden = self.activation(hidden)
        output_coord = self.to_coord(hidden)
        # fix boundary
        self.fix_boundary(output_coord)
        return output_coord, hidden

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data[:, 0] == 1
        self.down_node_idx = in_data[:, 0] == 0
        self.left_node_idx = in_data[:, 1] == 0
        self.right_node_idx = in_data[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1
