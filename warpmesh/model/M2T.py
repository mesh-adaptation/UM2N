# This file is not written by the author of the project.
# The purose of this file is for comparison with the MRN model.
# The impelemented DeformGAT class is from M2N paper:
# https://arxiv.org/abs/2204.11188
# The original code is from: https://github.com/erizmr/M2N. However,
# this is a private repo belongs to https://github.com/erizmr, So the
# marker of this project may need to contact the original author for
# original code base.

import sys
import os
import torch
import torch.nn.functional as F
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from gatdeformer import DeformGAT  # noqa: E402
from transformer_model import TransformerModel

__all__ = ['M2Transformer']


class NetGATDeform(torch.nn.Module):
    def __init__(self, in_dim):
        super(NetGATDeform, self).__init__()
        self.lin = torch.nn.Linear(in_dim, 254)
        self.gat_1 = DeformGAT(256, 508, heads=6)
        self.gat_2 = DeformGAT(512, 250, heads=6)
        self.gat_3 = DeformGAT(256, 120, heads=6)
        self.gat_4 = DeformGAT(128, 20, heads=6)

    def forward(self, data, edge_idx):
        coords_tensor = data[:, 0:2]
        lin_1 = self.lin(data)
        lin_1 = F.selu(lin_1)
        together_1 = torch.cat([coords_tensor, lin_1], dim=1)

        out_coord_1, out_feature_1 = self.gat_1(
            coords_tensor, together_1, edge_idx)

        together_2 = torch.cat(
            [out_coord_1, coords_tensor, out_feature_1], dim=1)
        out_coord_2, out_feature_2 = self.gat_2(
            out_coord_1, together_2, edge_idx)
        # 下面是第三层gat的准备层了啊。。
        together_3 = torch.cat(
            [out_coord_2, out_coord_1, coords_tensor, out_feature_2], dim=1)
        out_coord_3, out_feature_3 = self.gat_3(
            out_coord_2, together_3, edge_idx)
        # 下面是第四层gat的准备层了啊。。
        together_4 = torch.cat(
            [out_coord_3, out_coord_2, out_coord_1,
             coords_tensor, out_feature_3], dim=1)
        out_coord_4, out_feature_4 = self.gat_4(
            out_coord_3, together_4, edge_idx)

        return out_coord_4


class M2Transformer(torch.nn.Module):
    def __init__(self, gfe_in_c=1, lfe_in_c=3, deform_in_c=7, use_drop=False):
        super().__init__()
        self.gfe_out_c = 16
        self.lfe_out_c = 16
        self.deformer_in_feat = 7 + self.gfe_out_c + self.lfe_out_c

        self.deformer = NetGATDeform(in_dim=self.deformer_in_feat)

        self.transformer_in_dim = 3
        self.transformer_out_dim = 32
        self.transformer_encoder = TransformerModel(input_dim=self.transformer_in_dim, embed_dim=64, output_dim=self.transformer_out_dim, num_heads=4, num_layers=1)

    def forward(self, data):
        x = data.x  # [num_nodes * batch_size, 2]
        conv_feat_in = data.conv_feat_fix  # [batch_size, feat, 20, 20], using fixed conv-sample. # noqa
        batch_size = conv_feat_in.shape[0]
        mesh_feat = data.mesh_feat  # [num_nodes * batch_size, 2]
        edge_idx = data.edge_index  # [num_edges * batch_size, 2]
        node_num = data.node_num

        # conv_feat = self.gfe(conv_feat_in)
        # conv_feat = conv_feat.repeat_interleave(
        #     node_num.reshape(-1), dim=0)

        # local_feat = self.lfe(mesh_feat, edge_idx)
        feat_dim = mesh_feat.shape[-1]
        # mesh_feat [coord_x, coord_y, u, hessian_norm]
        features = self.transformer_encoder(mesh_feat.reshape(batch_size, -1, feat_dim))
        features = features.reshape(-1, self.transformer_out_dim)

        x = torch.cat([x, features], dim=1)
        # x = torch.cat([x, local_feat, conv_feat], dim=1)
        x = self.deformer(x, edge_idx)

        return x