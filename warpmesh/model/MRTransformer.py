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
    LocalFeatExtractor, GlobalFeatExtractor
)
from transformer_model import TransformerModel
__all__ = ['MRTransformer']


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


class MLPDeformer(MessagePassing):
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


class MRTransformer(torch.nn.Module):
    """
    Mesh Refinement Network (MRN) implementing global and local feature
        extraction and recurrent graph-based deformations. 
        The global feature extraction is performed by a transformer.

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
    def __init__(self, gfe_in_c=2, lfe_in_c=4,
                 deform_in_c=7, num_loop=3):
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
        # self.gfe_out_c = 16
        # self.lfe_out_c = 16
        self.hidden_size = 512  # set here
        # minus 2 because we are not using x,y coord (first 2 channels)
        # self.all_feat_c = (
        #     (deform_in_c-2) + self.gfe_out_c + self.lfe_out_c)

        self.transformer_in_dim = 4
        self.transformer_out_dim = 16
        self.transformer_encoder = TransformerModel(input_dim=self.transformer_in_dim, embed_dim=64, output_dim=self.transformer_out_dim, num_heads=4, num_layers=1)
        self.all_feat_c = (
            (deform_in_c-2) + self.transformer_out_dim)
        # use a linear layer to transform the input feature to hidden
        # state size
        self.lin = nn.Linear(self.all_feat_c, self.hidden_size)
        self.deformer = RecurrentGATConv(
            coord_size=2,
            hidden_size=self.hidden_size,
            heads=6,
            concat=False
        )

    def _forward(self, data):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        # coord = data.x[:, :2]  # [num_nodes * batch_size, 2]
        conv_feat_in = data.conv_feat  # [batch_size, feat, grid, grid]
        batch_size = conv_feat_in.shape[0]
        mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 4]
        feat_dim = mesh_feat.shape[-1]
        # mesh_feat [coord_x, coord_y, u, hessian_norm]
        features = self.transformer_encoder(mesh_feat.reshape(batch_size, -1, feat_dim))
        features = features.reshape(-1, self.transformer_out_dim)
        features = torch.cat([data.x[:, 2:], features], dim=1)
        features = F.selu(self.lin(features))
        return features

    def forward(self, data):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """

        coord = data.x[:, :2]
        edge_idx = data.edge_index
        hidden = self._forward(data)

        # Recurrent GAT deform
        for i in range(self.num_loop):
            coord, hidden = self.deformer(coord, hidden, edge_idx)

        return coord

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
        coord = data.x[:, :2]
        edge_idx = data.edge_index
        hidden = self._forward(data)

        # Recurrent GAT deform
        for i in range(num_step):
            coord, hidden = self.deformer(coord, hidden, edge_idx)

        return coord