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
    def __init__(self, num_transformer_in=4, 
                 num_transformer_out=16, 
                 num_transformer_embed_dim=64, 
                 num_transformer_heads=4, 
                 num_transformer_layers=1, 
                 transformer_training_mask=False, 
                 transformer_training_mask_ratio_lower_bound=0.5,
                 transformer_training_mask_ratio_upper_bound=0.9,
                 deform_in_c=7, num_loop=3, device='cuda'):
        """
        Initialize MRN.

        Args:
            gfe_in_c (int): Input channels for the global feature extractor.
            lfe_in_c (int): Input channels for the local feature extractor.
            deform_in_c (int): Input channels for the deformer block.
            num_loop (int): Number of loops for the recurrent layer.
        """
        super().__init__()
        self.device =device
        self.num_loop = num_loop
        self.hidden_size = 512  # set here
        self.mask_in_trainig = transformer_training_mask
        self.mask_ratio_ub = transformer_training_mask_ratio_upper_bound
        self.mask_ratio_lb = transformer_training_mask_ratio_lower_bound
        assert self.mask_ratio_ub >= self.mask_ratio_lb, 'Training mask ratio upper bound smaller than lower bound.'

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
    
    def _forward(self, batch_size, mesh_feat, x_feat, get_attens=False):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        # mesh_feat: [num_nodes * batch_size, 4]
        feat_dim = mesh_feat.shape[-1]
        # mesh_feat [coord_x, coord_y, u, hessian_norm]
        transformer_input = mesh_feat.reshape(batch_size, -1, feat_dim)
        node_num = transformer_input.shape[1]

        key_padding_mask = None
        attention_mask = None
        if self.train and self.mask_in_trainig:
            mask_ratio = (self.mask_ratio_ub - self.mask_ratio_lb) * torch.rand(1) + self.mask_ratio_lb
            masked_num = int(node_num * mask_ratio)
            mask = torch.randperm(node_num)[:masked_num]

            # Key padding mask
            # key_padding_mask = torch.zeros([batch_size, node_num], dtype=torch.bool).to(self.device)
            # key_padding_mask[:, mask] = True
            # print(key_padding_mask.shape, key_padding_mask)
            # print("Now is training")

            # Attention mask
            attention_mask = torch.zeros([batch_size*self.num_heads, node_num, node_num], dtype=torch.bool).to(self.device)
            attention_mask[:, mask, mask] = True
        
        features = self.transformer_encoder(transformer_input, key_padding_mask=key_padding_mask, attention_mask=attention_mask)
        features = features.reshape(-1, self.num_transformer_out)
        features = torch.cat([x_feat[:, 2:], features], dim=1)
        features = F.selu(self.lin(features))

        if not get_attens:
            return features
        else:
            atten_scores = self.transformer_encoder.get_attention_scores(x=transformer_input, key_padding_mask=key_padding_mask)
            return features, atten_scores
    
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
        conv_feat_in = data.conv_feat
        batch_size = batch_size = conv_feat_in.shape[0]
        feat_dim = data.x.shape[-1]
        x_feat = data.x.reshape(-1, feat_dim)
        coord = x_feat[:, :2]
        edge_idx = data.edge_index
        hidden = self._forward(batch_size, data.mesh_feat, x_feat)

        # Recurrent GAT deform
        for i in range(num_step):
            coord, hidden = self.deformer(coord, hidden, edge_idx)

        return coord

    def forward(self, data):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        conv_feat_in = data.conv_feat
        batch_size = batch_size = conv_feat_in.shape[0]
        feat_dim = data.x.shape[-1]
        x_feat = data.x.reshape(-1, feat_dim)
        coord = x_feat[:, :2]
        edge_idx = data.edge_index
        hidden = self._forward(batch_size, data.mesh_feat, x_feat)

        # Recurrent GAT deform
        for i in range(self.num_loop):
            coord, hidden = self.deformer(coord, hidden, edge_idx)

        return coord

    def get_attention_scores(self, data):
        conv_feat_in = data.conv_feat
        batch_size = batch_size = conv_feat_in.shape[0]
        feat_dim = data.x.shape[-1]
        x_feat = data.x.reshape(-1, feat_dim)
        # coord = x_feat[:, :2]
        # edge_idx = data.edge_index
        _, attentions = self._forward(batch_size, data.mesh_feat, x_feat, get_attens=True)
        return attentions

