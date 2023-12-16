# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# Modified by Mingrui Zhang

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)

from deformer import (  # noqa: E402
    RecurrentGATConv
)
from transformer_model import TransformerModel
__all__ = ['MRTransformer']


class MRTransformer(torch.nn.Module):
    """
    Mesh Refinement Network (MRN) implementing transformer as feature
        extrator and recurrent graph-based deformations. 

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
    def __init__(self, num_transformer_in=4, num_transformer_out=16, num_transformer_embed_dim=64, num_transformer_heads=4, num_transformer_layers=1,
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
        self.hidden_size = 512  # set here

        self.num_transformer_in = num_transformer_in
        self.num_transformer_out = num_transformer_out
        self.num_transformer_embed_dim = num_transformer_embed_dim
        self.num_heads = num_transformer_heads
        self.num_layers = num_transformer_layers
        self.transformer_encoder = TransformerModel(input_dim=self.num_transformer_in, embed_dim=self.num_transformer_embed_dim, output_dim=self.num_transformer_out, num_heads=self.num_heads, num_layers=self.num_layers)
        self.all_feat_c = (
            (deform_in_c-2) + self.num_transformer_out)
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
        features = features.reshape(-1, self.num_transformer_out)
        features = torch.cat([data.x[:, 2:], features], dim=1)
        features = F.selu(self.lin(features))
        return features

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
    #     coord = data.x[:, :2]
    #     edge_idx = data.edge_index
    #     hidden = self._forward(data)

    #     # Recurrent GAT deform
    #     for i in range(num_step):
    #         coord, hidden = self.deformer(coord, hidden, edge_idx)

    #     return coord

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
