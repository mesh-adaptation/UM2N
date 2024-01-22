# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# Modified by Mingrui Zhang

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
                 heads=6, output_type='coord',
                 concat=False, device='cuda'
                 ):
        super(RecurrentGATConv, self).__init__()
        assert output_type in ['coord', 'phi_grad', 'phi'], f"output type {output_type} is invalid"
        self.device = device
        self.output_type = output_type
        if self.output_type == 'coord' or self.output_type == 'phi_grad':
            self.output_dim = 2
        elif output_type == 'phi':
            self.output_dim = 1
        else:
            raise Exception(f"Output type {output_type} is invalid.")

        # GAT layer
        self.to_hidden = GATv2Conv(
            in_channels=coord_size+hidden_size,
            out_channels=hidden_size,
            heads=heads,
            concat=concat
        )
        # output layer
        self.to_output = nn.Sequential(
            nn.Linear(hidden_size, self.output_dim),
        )
        # activation function
        self.activation = nn.SELU()

    def forward(self, coord, hidden_state, edge_index, coord_ori):
        # if self.output_dim == 2:
        #     # find boundary
        #     self.find_boundary(coord)
        # data.mesh_feat.requires_grad = True
        # coord_ori = data.mesh_feat[:, :2]

        # Recurrent GAT
        in_feat = torch.cat((coord, hidden_state), dim=1)
        hidden = self.to_hidden(in_feat, edge_index)
        hidden = self.activation(hidden)
        output = self.to_output(hidden)
        phix = None
        phiy = None
        if self.output_type == 'coord':
            output_coord = output
            # find boundary
            self.find_boundary(coord_ori)
            # fix boundary
            self.fix_boundary(output_coord)
        elif self.output_type == 'phi_grad':
            output_coord = output + coord_ori
            # find boundary
            self.find_boundary(coord_ori)
            # fix boundary
            self.fix_boundary(output_coord)
            phix = output[:, 0]
            phiy = output[:, 1]
        elif self.output_type == 'phi':
            # Compute the residual to the equation
            grad_seed = torch.ones(output.shape).to(self.device)
            phi_grad = torch.autograd.grad(output, coord_ori, grad_outputs=grad_seed, retain_graph=True, create_graph=True, allow_unused=False)[0]
            # print(f"[phi grad] {phi_grad.shape}, [coord_ori] {coord_ori.shape}")
            phix = phi_grad[:, 0]
            phiy = phi_grad[:, 1]
            # New coord
            coord_x = (coord_ori[:, 0] + phix).reshape(-1, 1)
            coord_y = (coord_ori[:, 1] + phiy).reshape(-1, 1)
            output_coord = torch.cat([coord_x, coord_y], dim=-1).reshape(-1, 2)
            # find boundary
            self.find_boundary(coord_ori)
            # fix boundary
            self.fix_boundary(output_coord)
            # print('[phi] output coord shape ', output_coord.shape)
        return (output_coord, output), hidden, (phix, phiy)

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
