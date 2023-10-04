# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import sys
import os
from torch_geometric.nn import GATConv, GATv2Conv
import torch
import torch.nn.functional as F
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from extractor import (  # noqa
    LocalFeatExtractor, GlobalFeatExtractor
)


__all__ = ['GAT', 'NetGAT', 'NetGATFix',
           'NetGlobGATFix', 'NetCnnGATFixRecur', 'NetCnnGATv2Fix',
           'NetGlobLocalGATFix', 'WmDeformer',
           'WmDeformerBlock', 'NetGlobLocalGATFixDeformer',
           'GATDeformer', 'GATDeformerBlock',
           'GlobLocalGATDeformer', 'GlobLocalGATDeformerRes']


class GAT(torch.nn.Module):
    def __init__(self, in_feature, out_feature, heads=6):
        super().__init__()
        self.conv1 = GATConv(in_feature, out_feature,
                             heads=heads, concat=False)

    def forward(self, data, edge_idx):
        x = self.conv1(data, edge_idx)
        x = F.selu(x)
        return x


class GATv2(torch.nn.Module):
    def __init__(self, in_feature, out_feature, heads=6):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_feature, out_feature,
            heads=heads, concat=False)

    def forward(self, data, edge_idx):
        x = self.conv1(data, edge_idx)
        x = F.selu(x)
        return x


class GATDeformer(torch.nn.Module):
    def __init__(self, in_feature, out_feature, heads=6):
        super().__init__()
        self.alpha_coeff = 0.2
        self.gat_conv = GATConv(
            in_feature, out_feature,
            heads=heads, concat=False)

    def forward(self, data, edge_idx):
        self.find_boundary(data)
        feat, (edge_idx_, alpha) = self.gat_conv(
            data, edge_idx,
            return_attention_weights=True)
        alpha_scaled = alpha * self.alpha_coeff
        coord = data[:, :2]
        x_coord_l = x_coord_r = coord.unsqueeze(1)

        out_coord = self.gat_conv.propagate(
            edge_index=edge_idx_,
            x=(x_coord_l, x_coord_r),
            alpha=alpha_scaled,
        )
        out_coord = torch.mean(out_coord, dim=1)
        self.fix_boundary(out_coord)
        feat = F.selu(feat)
        return out_coord, feat

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


class GATDeformerBlock(torch.nn.Module):
    def __init__(self, in_dim=10):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, 256)
        self.activate = torch.nn.SELU()

        self.Deformer1 = GATDeformer(256 + 2, 512, heads=8)
        self.Deformer2 = GATDeformer(512 + 4, 256, heads=8)
        self.Deformer3 = GATDeformer(256 + 6, 128, heads=8)
        self.Deformer4 = GATDeformer(128 + 8, 20, heads=8)

    def forward(self, data, edge_idx):
        coord = data[:, :2]

        feat = self.lin(data)
        feat = self.activate(feat)

        all_feat1 = torch.cat([coord, feat], dim=1)  # 2 + 256 = 258
        coord1, feat1 = self.Deformer1(all_feat1, edge_idx)  # 2, 512

        all_feat2 = torch.cat(
            [coord1, coord, feat1], dim=1)  # 2 + 2 + 512 = 516
        coord2, feat2 = self.Deformer2(all_feat2, edge_idx)  # 2, 256

        all_feat3 = torch.cat(
            [coord2, coord1, coord, feat2], dim=1)  # 2 + 2 + 2 + 256 = 262
        coord3, feat3 = self.Deformer3(all_feat3, edge_idx)  # 2, 128

        all_feat4 = torch.cat(
            [coord3, coord2, coord1, coord, feat3],
            dim=1)  # 2 + 2 + 2 + 2 + 128 = 136

        coord4, _ = self.Deformer4(all_feat4, edge_idx)
        return coord4


class GATDeformeResrBlock(torch.nn.Module):
    def __init__(self, num_feat=10):
        super().__init__()
        self.lin = torch.nn.Linear(num_feat, 256)
        self.activate = torch.nn.SELU()

        self.Deformer1 = GATDeformer(258, 512, heads=6)
        self.Deformer2 = GATDeformer(516, 256, heads=6)
        self.Deformer3 = GATDeformer(262, 128, heads=6)
        self.Deformer4 = GATDeformer(136, 32, heads=6)

    def forward(self, data, edge_idx):
        coord = data[:, :2]

        feat = self.lin(data)
        feat = self.activate(feat)

        all_feat1 = torch.cat([coord, feat], dim=1)  # 2 + 256 = 258
        coord1, feat1 = self.Deformer1(all_feat1, edge_idx)  # 2, 512

        all_feat2 = torch.cat(
            [coord1, coord, feat1], dim=1)  # 2 + 2 + 512 = 516
        coord2, feat2 = self.Deformer2(all_feat2, edge_idx)  # 2, 256

        all_feat3 = torch.cat(
            [coord2, coord1, coord, feat2], dim=1)  # 2 + 2 + 2 + 256 = 262
        coord3, feat3 = self.Deformer3(all_feat3, edge_idx)  # 2, 128

        all_feat4 = torch.cat(
            [coord3, coord2, coord1, coord, feat3],
            dim=1)  # 2 + 2 + 2 + 2 + 128 = 136

        coord4, _ = self.Deformer4(all_feat4, edge_idx)

        out = coord + coord4

        return out


class WmDeformer(torch.nn.Module):
    def __init__(self, in_feature, out_feature, heads=6):
        super().__init__()
        self.feat_num = 2
        self.conv_feat = GATConv(
            in_feature, out_feature,
            heads=heads, concat=False)
        self.conv_coord = GATConv(
            in_feature, 2,
            heads=heads, concat=False)

    def forward(self, data, edge_idx):
        self.find_boundary(data)
        # coord = data[:, :self.feat_num]
        coord = self.conv_coord(data, edge_idx)

        feat = self.conv_feat(data, edge_idx)
        feat = F.selu(feat)
        self.fix_boundary(coord)
        return coord, feat

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


class WmDeformerBlock(torch.nn.Module):
    def __init__(self, num_feat=10):
        super().__init__()
        self.lin = torch.nn.Linear(num_feat, 256)
        self.activate = torch.nn.SELU()

        self.Deformer1 = WmDeformer(258, 512, heads=6)
        self.Deformer2 = WmDeformer(516, 256, heads=6)
        self.Deformer3 = WmDeformer(262, 128, heads=6)
        self.Deformer4 = WmDeformer(136, 32, heads=6)

    def forward(self, data, edge_idx):
        coord = data[:, :2]
        feat = data
        feat = self.lin(feat)
        feat = self.activate(feat)

        all_feat1 = torch.cat([coord, feat], dim=1)  # 2 + 256 = 258
        coord1, feat1 = self.Deformer1(all_feat1, edge_idx)  # 2, 512

        all_feat2 = torch.cat(
            [coord, coord1, feat1], dim=1)  # 2 + 2 + 512 = 516
        coord2, feat2 = self.Deformer2(all_feat2, edge_idx)  # 2, 256

        all_feat3 = torch.cat(
            [coord, coord1, coord2, feat2], dim=1)  # 2 + 2 + 2 + 256 = 262
        coord3, feat3 = self.Deformer3(all_feat3, edge_idx)  # 2, 128

        all_feat4 = torch.cat(
            [coord, coord1, coord2, coord3, feat3],
            dim=1)  # 2 + 2 + 2 + 2 + 128 = 136

        coord4, feat4 = self.Deformer4(all_feat4, edge_idx)
        return coord4


class NetGAT(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, 16)
        self.gat_1 = GAT(16, 32, heads=6)
        self.gat_2 = GAT(32, 32, heads=6)
        self.gat_3 = GAT(32, 16, heads=6)
        self.gat_4 = GAT(16, 2, heads=6)

    def forward(self, data):
        x, edge_idx = data.x, data.edge_index
        lin_1 = self.lin(x)
        lin_1 = F.selu(lin_1)
        x = lin_1
        out_coord_1 = self.gat_1(x, edge_idx)
        out_coord_2 = self.gat_2(out_coord_1, edge_idx)
        out_coord_3 = self.gat_3(out_coord_2, edge_idx)
        out_coord_4 = self.gat_4(out_coord_3, edge_idx)

        return out_coord_4


class NetGATFix(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, 16)
        self.gat_1 = GAT(16, 32, heads=6)
        self.gat_2 = GAT(32, 32, heads=6)
        self.gat_3 = GAT(32, 16, heads=6)
        self.gat_4 = GAT(16, 2, heads=6)

    def forward(self, data):
        self.find_boundary(data)

        x, edge_idx = data.x, data.edge_index
        lin_1 = self.lin(x)
        lin_1 = F.selu(lin_1)

        x = lin_1
        self.fix_boundary(x)

        x = self.gat_1(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_2(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_3(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_4(x, edge_idx)
        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class NetGlobGATFix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_out_c = 24
        self.Cnn = GlobalFeatExtractor(in_c=4, out_c=self.cnn_out_c)

        self.gat_1 = GAT(10 + self.cnn_out_c, 64, heads=8)
        self.gat_2 = GAT(64, 128, heads=16)
        self.gat_3 = GAT(128, 256, heads=8)
        self.gat_4 = GAT(256, 128, heads=8)
        self.gat_5 = GAT(128, 2, heads=16)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.Cnn(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.cnn_out_c)
        x = torch.cat([conv_feat, x], dim=1)

        x = self.gat_1(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_2(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_3(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_4(x, edge_idx)
        self.fix_boundary(x)

        res = self.gat_5(x, edge_idx)

        # use movement
        x = data.x[:, :2] + res
        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class NetCnnGATFixRecur(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_out_c = 16
        self.Cnn = NetGlobGATFix(in_c=4, out_c=self.cnn_out_c)

        self.gat_1 = GAT(10 + self.cnn_out_c, 32, heads=8)
        self.gat_2 = GAT(32, 64, heads=8)
        self.gat_3 = GAT(64, 32, heads=8)
        self.gat_4 = GAT(32, 10 + self.cnn_out_c, heads=8)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.Cnn(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.cnn_out_c)
        x = torch.cat([conv_feat, x], dim=1)
        recur_num = 3
        for i in range(recur_num):
            x = self.recur(data, x, edge_idx)
        return x[:, :2]

    def recur(self, data, x, edge_idx):
        x = self.gat_1(x, edge_idx)
        # self.fix_boundary(x)

        x = self.gat_2(x, edge_idx)
        # self.fix_boundary(x)

        x = self.gat_3(x, edge_idx)
        # self.fix_boundary(x)

        res = self.gat_4(x, edge_idx)

        # use movement
        res[:, :2] = data.x[:, :2] + res[:, :2]

        x = res

        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class NetCnnGATv2Fix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_out_c = 24
        self.Cnn = GlobalFeatExtractor(in_c=4, out_c=self.cnn_out_c)

        self.gat_1 = GATv2(10 + self.cnn_out_c, 64, heads=8)
        self.gat_2 = GATv2(64, 128, heads=16)
        self.gat_3 = GATv2(128, 256, heads=8)
        self.gat_4 = GATv2(256, 128, heads=8)
        self.gat_5 = GATv2(128, 2, heads=16)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.Cnn(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.cnn_out_c)
        x = torch.cat([conv_feat, x], dim=1)

        x = self.gat_1(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_2(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_3(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_4(x, edge_idx)
        self.fix_boundary(x)

        res = self.gat_5(x, edge_idx)

        # use movement
        x = data.x[:, :2] + res
        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class NetGlobLocalGATFix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gfe_out_c = 16
        self.lfe_out_c = 16

        self.gfe = GlobalFeatExtractor(in_c=4, out_c=self.gfe_out_c)
        self.lfe = LocalFeatExtractor(num_feat=10, out=self.lfe_out_c)

        self.gat_1 = GAT(10 + self.gfe_out_c + self.lfe_out_c,
                         64, heads=8)
        self.gat_2 = GAT(64, 128, heads=16)
        self.gat_3 = GAT(128, 256, heads=8)
        self.gat_4 = GAT(256, 128, heads=8)
        self.gat_5 = GAT(128, 2, heads=16)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.gfe_out_c)

        local_feat = self.lfe(x, edge_idx)

        x = torch.cat([conv_feat, local_feat, x], dim=1)

        x = self.gat_1(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_2(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_3(x, edge_idx)
        self.fix_boundary(x)

        x = self.gat_4(x, edge_idx)
        self.fix_boundary(x)

        res = self.gat_5(x, edge_idx)

        # use movement
        x = data.x[:, :2] + res
        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class NetGlobLocalGATFixDeformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gfe_out_c = 16
        self.lfe_out_c = 16
        self.deformer_in_feat = 10 + self.gfe_out_c + self.lfe_out_c

        self.gfe = GlobalFeatExtractor(in_c=4, out_c=self.gfe_out_c)
        self.lfe = LocalFeatExtractor(num_feat=10, out=self.lfe_out_c)
        self.deformer = WmDeformerBlock(num_feat=self.deformer_in_feat)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.gfe_out_c)

        local_feat = self.lfe(x, edge_idx)

        x = torch.cat([conv_feat, local_feat, x], dim=1)

        x = self.deformer(x, edge_idx)

        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class GlobLocalGATDeformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gfe_out_c = 32
        self.lfe_out_c = 32
        self.deformer_in_feat = 10 + self.gfe_out_c + self.lfe_out_c

        self.gfe = GlobalFeatExtractor(in_c=4, out_c=self.gfe_out_c)
        self.lfe = LocalFeatExtractor(num_feat=10, out=self.lfe_out_c)
        self.deformer = GATDeformerBlock(num_feat=self.deformer_in_feat)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.gfe_out_c)

        local_feat = self.lfe(x, edge_idx)

        x = torch.cat([x, local_feat, conv_feat], dim=1)

        x = self.deformer(x, edge_idx)

        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1


class GlobLocalGATDeformerRes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gfe_out_c = 32
        self.lfe_out_c = 32
        self.deformer_in_feat = 10 + self.gfe_out_c + self.lfe_out_c

        self.gfe = GlobalFeatExtractor(in_c=4, out_c=self.gfe_out_c)
        self.lfe = LocalFeatExtractor(num_feat=10, out=self.lfe_out_c)
        self.deformer = GATDeformeResrBlock(num_feat=self.deformer_in_feat)

    def forward(self, data):
        self.find_boundary(data)

        x = data.x
        edge_idx = data.edge_index
        conv_feat_in = data.conv_feat

        num_nodes = x.shape[0]
        conv_feat = self.gfe(conv_feat_in)
        conv_feat = conv_feat.expand(num_nodes, self.gfe_out_c)

        local_feat = self.lfe(x, edge_idx)

        x = torch.cat([conv_feat, local_feat, x], dim=1)

        x = self.deformer(x, edge_idx)

        self.fix_boundary(x)

        return x

    def find_boundary(self, in_data):
        self.upper_node_idx = in_data.x[:, 0] == 1
        self.down_node_idx = in_data.x[:, 0] == 0
        self.left_node_idx = in_data.x[:, 1] == 0
        self.right_node_idx = in_data.x[:, 1] == 1

    def fix_boundary(self, in_data):
        in_data[self.upper_node_idx, 0] = 1
        in_data[self.down_node_idx, 0] = 0
        in_data[self.left_node_idx, 1] = 0
        in_data[self.right_node_idx, 1] = 1
