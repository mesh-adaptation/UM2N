# Author: Chunyang Wang
# GitHub Username: acse-cw1722

from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from transformer_model import TransformerModel

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
        # print("x_i x_j ", x_i.shape, x_j.shape, "x_edge_feat ", x_edge_feat.shape)

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
        print(f"before selu {x.shape}")
        x = F.selu(x)
        x = self.dropout(x) if self.use_drop else x
        x = self.final_pool(x)
        print(f"after final pool {x.shape}")
        x = x.reshape(-1, self.out_c)
        return x


class TransformerEncoder(torch.nn.Module):
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

    def __init__(
        self,
        num_transformer_in=3,
        num_transformer_out=16,
        num_transformer_embed_dim=64,
        num_transformer_heads=4,
        num_transformer_layers=1,
        transformer_training_mask=False,
        transformer_key_padding_training_mask=False,
        transformer_attention_training_mask=False,
        transformer_training_mask_ratio_lower_bound=0.5,
        transformer_training_mask_ratio_upper_bound=0.9,
        deform_in_c=7,
        deform_out_type="coord",
        num_loop=3,
        device="cuda",
    ):
        """
        Initialize MRN.

        Args:
            gfe_in_c (int): Input channels for the global feature extractor.
            lfe_in_c (int): Input channels for the local feature extractor.
            deform_in_c (int): Input channels for the deformer block.
            num_loop (int): Number of loops for the recurrent layer.
        """
        super().__init__()
        self.device = device
        # self.num_loop = num_loop
        # self.hidden_size = 512  # set here
        self.mask_in_trainig = transformer_training_mask
        self.key_padding_mask_in_training = transformer_key_padding_training_mask
        self.attention_mask_in_training = transformer_attention_training_mask
        self.mask_ratio_ub = transformer_training_mask_ratio_upper_bound
        self.mask_ratio_lb = transformer_training_mask_ratio_lower_bound
        assert (
            self.mask_ratio_ub >= self.mask_ratio_lb
        ), "Training mask ratio upper bound smaller than lower bound."

        self.num_transformer_in = num_transformer_in
        self.num_transformer_out = num_transformer_out
        self.num_transformer_embed_dim = num_transformer_embed_dim
        self.num_heads = num_transformer_heads
        self.num_layers = num_transformer_layers
        self.transformer_encoder = TransformerModel(
            input_dim=self.num_transformer_in,
            embed_dim=self.num_transformer_embed_dim,
            output_dim=self.num_transformer_out,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        )
        # self.all_feat_c = (deform_in_c - 2) + self.num_transformer_out
        # use a linear layer to transform the input feature to hidden
        # state size
        # self.lin = nn.Linear(self.all_feat_c, self.hidden_size)

    def _transformer_forward(
        self, batch_size, input_q, input_kv, get_attens=False
    ):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        # mesh_feat: [num_nodes * batch_size, 4]
        # mesh_feat [coord_x, coord_y, u, hessian_norm]
        transformer_input_q = input_q.view(batch_size, -1, input_q.shape[-1])
        transformer_input_kv = input_kv.view(batch_size, -1, input_kv.shape[-1])
        node_num = transformer_input_q.shape[1]
        # print("transformer input shape ", transformer_input_q.shape, transformer_input_kv.shape)

        key_padding_mask = None
        attention_mask = None
        if self.train and self.mask_in_trainig:
            mask_ratio = (self.mask_ratio_ub - self.mask_ratio_lb) * torch.rand(
                1
            ) + self.mask_ratio_lb
            masked_num = int(node_num * mask_ratio)
            mask = torch.randperm(node_num)[:masked_num]

            if self.key_padding_mask_in_training:
                # Key padding mask
                key_padding_mask = torch.zeros(
                    [batch_size, node_num], dtype=torch.bool
                ).to(self.device)
                key_padding_mask[:, mask] = True
            # print(key_padding_mask.shape, key_padding_mask)
            # print("Now is training")
            elif self.attention_mask_in_training:
                # Attention mask
                attention_mask = torch.zeros(
                    [batch_size * self.num_heads, node_num, node_num], dtype=torch.bool
                ).to(self.device)
                attention_mask[:, mask, mask] = True

        features = self.transformer_encoder(
            transformer_input_q,
            transformer_input_kv,
            transformer_input_kv,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        # features = features.view(-1, self.num_transformer_out)
        # features = torch.cat([boundary, features], dim=1)
        # print(f"transformer raw features: {features.shape}")
        features = F.selu(features)

        if not get_attens:
            return features
        else:
            # TODO: adapt q k v
            atten_scores = self.transformer_encoder.get_attention_scores(
                x=transformer_input_q, key_padding_mask=key_padding_mask
            )
            return features, atten_scores

    def forward(self, data):
        # batch_size = data.conv_feat.shape[0]
        batch_size = data.shape[0]
        feat_dim = data.shape[-1]
        # input_q, input_kv, boundary
        input_q = data.view(-1, feat_dim)
        input_kv = data.view(-1, feat_dim)

        # [coord_ori_x, coord_ori_y, u, hessian_norm]
        # intput_features = torch.cat([coord_ori, data.mesh_feat[:, 2:4]], dim=-1)
        # print(f"input q shape: {input_q.shape} input kv shape: {input_kv.shape}")
        hidden = self._transformer_forward(batch_size, input_q, input_kv)
        # print(f"global feat before reshape: {hidden.shape}")

        feat_dim = hidden.shape[-1]
        return hidden.view(-1, feat_dim)