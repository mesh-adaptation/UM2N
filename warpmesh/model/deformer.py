# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# Modified by Mingrui Zhang

import os
import sys

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, MessagePassing

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)

__all__ = ["RecurrentGATConv"]


class RecurrentGATConv(MessagePassing):
    """
    Implements a Recurrent Graph Attention Network (GAT) Convolution layer.

    Attributes:
        to_hidden (GATv2Conv): Graph Attention layer.
        to_coord (nn.Sequential): Output layer for coordinates.
        activation (nn.SELU): Activation function.
    """

    def __init__(
        self,
        coord_size=2,
        hidden_size=512,
        heads=6,
        output_type="coord",
        concat=False,
        device="cuda",
    ):
        super(RecurrentGATConv, self).__init__()
        assert output_type in [
            "coord",
            "phi_grad",
            "phi",
        ], f"output type {output_type} is invalid"
        self.device = device
        self.output_type = output_type
        if self.output_type == "coord" or self.output_type == "phi_grad":
            self.output_dim = 2
        elif output_type == "phi":
            self.output_dim = 1
        else:
            raise Exception(f"Output type {output_type} is invalid.")

        # GAT layer
        self.to_hidden = GATv2Conv(
            in_channels=coord_size + hidden_size,
            out_channels=hidden_size,
            heads=heads,
            concat=concat,
        )
        # output layer
        self.to_output = nn.Sequential(
            nn.Linear(hidden_size, self.output_dim),
        )
        # activation function
        self.activation = nn.SELU()

    def forward(self, coord, hidden_state, edge_index, coord_ori, bd_mask, poly_mesh):
        self.bd_mask = bd_mask.squeeze().bool()
        self.poly_mesh = poly_mesh

        # Recurrent GAT
        # print(coord.shape, hidden_state.shape)
        extra_sample_ratio = coord.shape[0] // hidden_state.shape[0]
        # print(coord.shape, hidden_state.shape)

        in_feat = torch.cat((coord, hidden_state.repeat(extra_sample_ratio, 1)), dim=1)
        # in_feat = torch.cat((coord, hidden_state), dim=1)
        hidden = self.to_hidden(in_feat, edge_index)
        hidden = self.activation(hidden)
        output = self.to_output(hidden)
        phix = None
        phiy = None
        if self.output_type == "coord":
            output_coord = output
            # find boundary
            self.find_boundary(coord_ori)
            # fix boundary
            self.fix_boundary(output_coord)
        elif self.output_type == "phi_grad":
            output_coord = output + coord_ori
            # find boundary
            self.find_boundary(coord_ori)
            # fix boundary
            self.fix_boundary(output_coord)
            phix = output[:, 0].view(-1, 1)
            phiy = output[:, 1].view(-1, 1)
        elif self.output_type == "phi":
            # Compute the residual to the equation
            grad_seed = torch.ones(output.shape).to(self.device)
            phi_grad = torch.autograd.grad(
                output,
                coord_ori,
                grad_outputs=grad_seed,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )[0]
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

        if self.poly_mesh:
            self.bd_pos_x = in_data[self.bd_mask, 0].clone()
            self.bd_pos_y = in_data[self.bd_mask, 1].clone()

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1

        if self.poly_mesh:
            in_data[self.bd_mask, 0] = self.bd_pos_x
            in_data[self.bd_mask, 1] = self.bd_pos_y
