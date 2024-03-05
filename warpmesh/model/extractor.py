# Author: Chunyang Wang
# GitHub Username: acse-cw1722

from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F

__all__ = ["LocalFeatExtractor", "GlobalFeatExtractor"]


class LocalFeatExtractor(MessagePassing):
    """
    Custom PyTorch geometric layer that performs feature extraction
    on local graph structure.

    The class extends the torch_geometric.nn.MessagePassing class
    and employs additive aggregation as the message-passing scheme.

    Attributes:
        lin_1 (torch.nn.Linear): First linear layer.
        lin_2 (torch.nn.Linear): Second linear layer.
        lin_3 (torch.nn.Linear): Third linear layer.
        activate (torch.nn.SELU): Activation function.
    """

    def __init__(self, num_feat=10, out=16):
        """
        Initialize the layer.

        Args:
            num_feat (int): Number of input features per node.
            out (int): Number of output features per node.
        """
        super().__init__(aggr="add")
        # 1*distance + 2*feat + 2*coord
        num_in_feat = 1 + (num_feat - 2) * 2 + 2
        self.lin_1 = torch.nn.Linear(num_in_feat, 64)
        self.lin_2 = torch.nn.Linear(64, 64)
        # minus 3 because dist, corrd is added back
        self.lin_3 = torch.nn.Linear(64, out - 1)
        self.activate = torch.nn.SELU()

    def forward(self, input, edge_index):
        """
        Forward pass.

        Args:
            input (Tensor): Node features.
            edge_index (Tensor): Edge indices.

        Returns:
            Tensor: Updated node features.
        """
        local_feat = self.propagate(edge_index, x=input)
        return local_feat

    def message(self, x_i, x_j):
        coord_idx = 2
        x_i_coord = x_i[:, :coord_idx]
        x_j_coord = x_j[:, :coord_idx]
        x_i_feat = x_i[:, coord_idx:]
        x_j_feat = x_j[:, coord_idx:]
        x_coord_diff = x_j_coord - x_i_coord
        x_coord_dist = torch.norm(x_coord_diff, dim=1, keepdim=True)

        x_edge_feat = torch.cat(
            [x_coord_diff, x_coord_dist, x_i_feat, x_j_feat], dim=1
        )  # [num_node, feat_dim]
        print("x_i x_j ", x_i.shape, x_j.shape, "x_edge_feat ", x_edge_feat.shape)

        x_edge_feat = self.lin_1(x_edge_feat)
        x_edge_feat = self.activate(x_edge_feat)
        x_edge_feat = self.lin_2(x_edge_feat)
        x_edge_feat = self.activate(x_edge_feat)
        x_edge_feat = self.lin_3(x_edge_feat)
        x_edge_feat = self.activate(x_edge_feat)

        x_edge_feat = torch.cat([x_edge_feat, x_coord_dist], dim=1)

        return x_edge_feat


class GlobalFeatExtractor(torch.nn.Module):
    """
    Custom PyTorch layer for global feature extraction.

    The class employs multiple convolutional layers and dropout layers.

    Attributes:
        conv1, conv2, conv3, conv4 (torch.nn.Conv2d): Convolutional layers.
        dropout (torch.nn.Dropout): Dropout layer.
        final_pool (torch.nn.AdaptiveAvgPool2d): Final pooling layer.
    """

    def __init__(self, in_c, out_c, drop_p=0.2, use_drop=True):
        super().__init__()
        """
        Initialize the layer.

        Args:
            in_c (int): Number of input channels.
            out_c (int): Number of output channels.
            drop_p (float, optional): Dropout probability.
            use_drop: (bool, optional): Use dropout layer or not,
                When it is set to `False`, this building block is exactly
                the block used in the original M2N model with out any
                change. Then set to `True`, it is used for MRN model.
        """
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = torch.nn.Conv2d(in_c, 32, 3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 32, 3, padding=2, stride=1)
        self.conv4 = torch.nn.Conv2d(32, out_c, 3, padding=2, stride=1)
        self.use_drop = use_drop
        self.dropout = torch.nn.Dropout(drop_p) if use_drop else None
        self.final_pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, data):
        """
        Forward pass.

        Args:
            data (Tensor): Input data.

        Returns:
            Tensor: Extracted global features.
        """
        x = self.conv1(data)
        x = F.selu(x)
        x = self.dropout(x) if self.use_drop else x
        x = self.conv2(x)
        x = F.selu(x)
        x = self.dropout(x) if self.use_drop else x
        x = self.conv3(x)
        x = F.selu(x)
        x = self.dropout(x) if self.use_drop else x
        x = self.conv4(x)
        x = F.selu(x)
        x = self.dropout(x) if self.use_drop else x
        x = self.final_pool(x)
        x = x.reshape(-1, self.out_c)
        return x
