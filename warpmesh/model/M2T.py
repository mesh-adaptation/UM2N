import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)

from M2T_deformer import M2TDeformer  # noqa: E402
from transformer_model import TransformerModel

__all__ = ["M2T"]


class M2T(torch.nn.Module):
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
        num_transformer_in=4,
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
        self.num_loop = num_loop
        self.hidden_size = 512  # set here
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
        self.all_feat_c = deform_in_c - 2 + self.num_transformer_out
        # use a linear layer to transform the input feature to hidden
        # state size
        self.lin = nn.Linear(self.all_feat_c, self.hidden_size)

        # Mapping embedding to monitor
        self.to_monitor_1 = nn.Linear(self.hidden_size, self.hidden_size // 8)
        self.to_monitor_2 = nn.Linear(self.hidden_size // 8, self.hidden_size // 16)
        self.to_monitor_3 = nn.Linear(self.hidden_size // 16, 1)

        self.deformer = M2TDeformer(
            feature_in_dim=self.all_feat_c,
            coord_size=2,
            hidden_size=self.hidden_size,
            heads=6,
            concat=False,
            output_type=deform_out_type,
            device=device,
        )

    def _transformer_forward(
        self, batch_size, input_q, input_kv, boundary, get_attens=False
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
        # print(transformer_input_q.shape, transformer_input_kv.shape)

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
        features = features.view(-1, self.num_transformer_out)
        features = torch.cat([boundary, features], dim=1)
        # print(f"transformer raw features: {features.shape}")
        features = F.selu(self.lin(features))

        if not get_attens:
            return features
        else:
            # TODO: adapt q k v
            atten_scores = self.transformer_encoder.get_attention_scores(
                x=transformer_input_q, key_padding_mask=key_padding_mask
            )
            return features, atten_scores

    def transformer_monitor(self, data, input_q, input_kv, boundary):
        batch_size = data.conv_feat.shape[0]

        # [coord_ori_x, coord_ori_y, u, hessian_norm]
        # intput_features = torch.cat([coord_ori, data.mesh_feat[:, 2:4]], dim=-1)
        # print(f"input q shape: {input_q.shape} input kv shape: {input_kv.shape}")
        hidden = self._transformer_forward(batch_size, input_q, input_kv, boundary)
        return hidden

    def forward(
        self,
        data,
        input_q,
        input_kv,
        mesh_query,
        sampled_queries,
        sampled_queries_edge_index,
        poly_mesh=False,
    ):
        """
        Forward pass for MRN.

        Args:
            data (Data): Input data object containing mesh and feature info.

        Returns:
            coord (Tensor): Deformed coordinates.
        """
        bd_mask = data.bd_mask
        poly_mesh = False
        if data.poly_mesh is not False:
            poly_mesh = True if data.poly_mesh.sum() > 0 else False

        edge_idx = data.edge_index

        boundary = data.x[:, 2:].view(-1, 1)
        hidden = self.transformer_monitor(data, input_q, input_kv, boundary)
        coord = mesh_query

        model_output = None
        out_monitor = None

        (coord, model_output), (phix, phiy) = self.deformer(
            coord, hidden, edge_idx, mesh_query, bd_mask, poly_mesh
        )
        if sampled_queries is not None:
            coord_extra = sampled_queries
            (coord_extra, model_output_extra), (phix_extra, phiy_extra) = self.deformer(
                coord_extra,
                hidden,
                sampled_queries_edge_index,
                sampled_queries,
                bd_mask,
                poly_mesh,
            )

            coord_output = torch.cat([coord, coord_extra], dim=0)
            model_raw_output = torch.cat([model_output, model_output_extra], dim=0)
            # # phix_output = torch.cat([phix, phix_extra], dim=0)
            # # phiy_output = torch.cat([phiy, phiy_extra], dim=0)
            phix_output = phix_extra
            phiy_output = phiy_extra
            # print(phix.shape, phix_extra.shape)
        else:
            coord_output = coord
            model_raw_output = model_output
            phix_output = phix
            phiy_output = phiy
        return (coord_output, model_raw_output, out_monitor), (phix_output, phiy_output)
        # return (coord, model_output, out_monitor), (phix, phiy)

    def get_attention_scores(self, data):
        conv_feat_in = data.conv_feat
        batch_size = batch_size = conv_feat_in.shape[0]
        feat_dim = data.x.shape[-1]
        x_feat = data.x.view(-1, feat_dim)
        # coord = x_feat[:, :2]
        # edge_idx = data.edge_index
        _, attentions = self._transformer_forward(
            batch_size, data.mesh_feat[:, :4], x_feat, get_attens=True
        )
        return attentions
