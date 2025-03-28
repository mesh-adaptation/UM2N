# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, MessagePassing

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from extractor import (  # noqa: E402
    GlobalFeatExtractor,
    LocalFeatExtractor,
)

__all__ = ["MRN", "RecurrentGATConv"]


class RecurrentGATConv(MessagePassing):
    """
    Implements a Recurrent Graph Attention Network (GAT) Convolution layer.

    Attributes:
        to_hidden (GATv2Conv): Graph Attention layer.
        to_coord (nn.Sequential): Output layer for coordinates.
        activation (nn.SELU): Activation function.
    """

    def __init__(self, coord_size=2, hidden_size=512, heads=6, concat=False):
        super(RecurrentGATConv, self).__init__()
        # GAT layer
        self.to_hidden = GATv2Conv(
            in_channels=coord_size + hidden_size,
            out_channels=hidden_size,
            heads=heads,
            concat=concat,
        )
        # output coord layer
        self.to_coord = nn.Sequential(
            nn.Linear(hidden_size, 2),
        )
        # activation function
        self.activation = nn.SELU()

    def forward(self, coord, hidden_state, edge_index, bd_mask, poly_mesh):
        # find boundary
        self.bd_mask = bd_mask.squeeze().bool()
        self.poly_mesh = poly_mesh
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


class MRN(torch.nn.Module):
    """
    Mesh Refinement Network (MRN) implementing global and local feature
        extraction
    and recurrent graph-based deformations.

    Attributes:
        num_loop (int): Number of loops for the recurrent layer.
        gfe_out_c (int): Output channels for global feature extractor.
        lfe_out_c (int): Output channels for local feature extractor.
        hidden_size (int): Size of the hidden layer.
        gfe (GlobalFeatExtractor): Global feature extractor.
        lfe (LocalFeatExtractor): Local feature extractor.
        lin (nn.Linear): Linear layer for feature transformation.
        deformer (RecurrentGATConv): GAT-based deformer block.
    """

    def __init__(self, gfe_in_c=2, lfe_in_c=4, deform_in_c=7, num_loop=3):
        """
        Initialize MRN.

        Args:
            gfe_in_c (int): Input channels for the global feature extractor.
            lfe_in_c (int): Input channels for the local feature extractor.
            deform_in_c (int): Input channels for the deformer block.
            num_loop (int): Number of loops for the recurrent layer.
        """
        super().__init__()
        self.num_loop = num_loop
        self.gfe_out_c = 16
        self.lfe_out_c = 16
        self.hidden_size = 512  # set here
        # minus 2 because we are not using x,y coord (first 2 channels)
        self.all_feat_c = (deform_in_c - 2) + self.gfe_out_c + self.lfe_out_c

        self.gfe = GlobalFeatExtractor(in_c=gfe_in_c, out_c=self.gfe_out_c)
        self.lfe = LocalFeatExtractor(num_feat=lfe_in_c, out=self.lfe_out_c)
        # use a linear layer to transform the input feature to hidden
        # state size
        self.lin = nn.Linear(self.all_feat_c, self.hidden_size)
        self.deformer = RecurrentGATConv(
            coord_size=2, hidden_size=self.hidden_size, heads=6, concat=False
        )
        # self.deformer = GATDeformerBlock(in_dim=self.deformer_in_feat)

    def move(self, data, num_step=1):
        """
        Move the mesh according to the deformation learned, with given number
            steps.

        Args:
            data (Data): Input data object containing mesh and feature info.
            num_step (int): Number of deformation steps.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        coord = data.x[:, :2]  # [num_nodes * batch_size, 2]
        conv_feat_in = data.conv_feat  # [batch_size, feat, grid, grid]
        mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 2]
        edge_idx = data.edge_index  # [num_edges * batch_size, 2]
        node_num = data.node_num

        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.repeat_interleave(node_num.reshape(-1), dim=0)

        local_feat = self.lfe(mesh_feat, edge_idx)

        hidden_in = torch.cat([data.x[:, 2:], local_feat, conv_feat], dim=1)
        hidden = F.selu(self.lin(hidden_in))

        # Recurrent GAT deform
        for i in range(num_step):
            coord, hidden = self.deformer(coord, hidden, edge_idx)

        return coord

    def forward(self, data, poly_mesh=False):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        bd_mask = data.bd_mask
        if data.poly_mesh is not False:
            poly_mesh = True if data.poly_mesh.sum() > 0 else False

        coord = data.x[:, :2]  # [num_nodes * batch_size, 2]
        conv_feat_in = data.conv_feat  # [batch_size, feat, grid, grid]
        mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 2]
        edge_idx = data.edge_index  # [num_edges * batch_size, 2]
        node_num = data.node_num

        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.repeat_interleave(node_num.reshape(-1), dim=0)

        local_feat = self.lfe(mesh_feat, edge_idx)

        hidden_in = torch.cat([data.x[:, 2:], local_feat, conv_feat], dim=1)
        hidden = F.selu(self.lin(hidden_in))

        # Recurrent GAT deform
        for i in range(self.num_loop):
            coord, hidden = self.deformer(coord, hidden, edge_idx, bd_mask, poly_mesh)  # noqa

        return coord
