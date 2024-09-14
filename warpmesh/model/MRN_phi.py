# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import sys
import os
from torch_geometric.nn import GATv2Conv, MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from extractor import (  # noqa: E402
    LocalFeatExtractor,
    GlobalFeatExtractor,
)

__all__ = ["MRN_phi", "RecurrentGATConv"]


class RecurrentGATConv(MessagePassing):
    """
    Implements a Recurrent Graph Attention Network (GAT) Convolution layer.

    Attributes:
        to_hidden (GATv2Conv): Graph Attention layer.
        to_coord (nn.Sequential): Output layer for coordinates.
        activation (nn.SELU): Activation function.
    """

    def __init__(self, phi_size=1, hidden_size=512, heads=6, concat=False):
        super(RecurrentGATConv, self).__init__()
        # GAT layer
        self.to_hidden = GATv2Conv(
            in_channels=phi_size + hidden_size,
            out_channels=hidden_size,
            heads=heads,
            concat=concat,
        )
        # output coord layer
        self.to_phi = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        # activation function
        self.activation = nn.SELU()

    def forward(self, phi, hidden_state, edge_index):
        # find boundary
        # Recurrent GAT
        in_feat = torch.cat((phi, hidden_state), dim=1)
        hidden = self.to_hidden(in_feat, edge_index)
        hidden = self.activation(hidden)
        output_phi = self.to_phi(hidden)

        return output_phi, hidden


class MRN_phi(torch.nn.Module):
    """
    Mesh Recurrent Network (MRN) implementing global and local feature
        extraction
    and recurrent graph-based deformations for field phi.

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
        self.all_feat_c = (deform_in_c) + self.gfe_out_c + self.lfe_out_c

        self.gfe = GlobalFeatExtractor(in_c=gfe_in_c, out_c=self.gfe_out_c)
        self.lfe = LocalFeatExtractor(num_feat=lfe_in_c, out=self.lfe_out_c)
        # use a linear layer to transform the input feature to hidden
        # state size
        self.lin = nn.Linear(self.all_feat_c, self.hidden_size)
        self.deformer = RecurrentGATConv(
            phi_size=1, hidden_size=self.hidden_size, heads=6, concat=False
        )

    def forward(self, data):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        # Initialize phi output to be zero
        # phi = torch.zeros_like(data.phi)  # [num_nodes * batch_size, 1]
        phi = data.mesh_feat[:, -1].reshape(-1, 1)  # [num_nodes * batch_size, 1]
        conv_feat_in = data.conv_feat  # [batch_size, feat, grid, grid]
        mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 2]
        edge_idx = data.edge_index  # [num_edges * batch_size, 2]
        node_num = data.node_num

        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.repeat_interleave(node_num.reshape(-1), dim=0)

        local_feat = self.lfe(mesh_feat, edge_idx)

        hidden_in = torch.cat([data.x, local_feat, conv_feat], dim=1)
        hidden = F.selu(self.lin(hidden_in))

        # Recurrent GAT deform
        for i in range(self.num_loop):
            phi, hidden = self.deformer(phi, hidden, edge_idx)

        return phi

    # def move(self, data, num_step=1):
    #     """
    #     Move the mesh according to the deformation learned, with given number
    #         steps.

    #     Args:
    #         data (Data): Input data object containing mesh and feature info.
    #         num_step (int): Number of deformation steps.

    #     Returns:
    #         coord (Tensor): Deformed coordinates.
    #     """
    #     coord = data.x[:, :2]  # [num_nodes * batch_size, 2]
    #     conv_feat_in = data.conv_feat  # [batch_size, feat, grid, grid]
    #     mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 2]
    #     edge_idx = data.edge_index  # [num_edges * batch_size, 2]
    #     node_num = data.node_num

    #     conv_feat = self.gfe(conv_feat_in)
    #     conv_feat = conv_feat.repeat_interleave(
    #         node_num.reshape(-1), dim=0)

    #     local_feat = self.lfe(mesh_feat, edge_idx)

    #     hidden_in = torch.cat(
    #         [data.x[:, 2:], local_feat, conv_feat], dim=1)
    #     hidden = F.selu(self.lin(hidden_in))

    #     # Recurrent GAT deform
    #     for i in range(num_step):
    #         phi, hidden = self.deformer(coord, hidden, edge_idx)

    #     return coord
