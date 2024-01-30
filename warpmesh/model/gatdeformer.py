# This file is not written by the author of the project.
# The purose of this file is for comparison with the MRN model.
# The impelemented DeformGAT class is from M2N paper:
# https://arxiv.org/abs/2204.11188
# The original code is from: https://github.com/erizmr/M2N. However,
# this is a private repo belongs to https://github.com/erizmr, So the
# marker of this project may need to contact the original author for
# original code base.

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import (
    OptPairTensor, Adj, OptTensor
)
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from typing import Union, Optional

__all__ = ['DeformGAT']


class DeformGAT(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = False,
                 negative_slope: float = 0.2,
                 dropout: float = 0,
                 bias: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(DeformGAT, self).__init__(node_dim=0, **kwargs)
        # comment：指定一些参数。。
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = False

        # comment:这边没有bias，我觉得不太行！！！
        # TODO：这里 bias 是 True 还是 False，再仔细想想看吧。
        self.lin_l = Linear(
            in_channels, heads * out_channels, bias=True).float()
        self.lin_ = self.lin_l

        # 这个是用来算attention的 vector
        self.att_l = Parameter(torch.FloatTensor(1, heads, out_channels))
        self.att_r = Parameter(torch.FloatTensor(1, heads, out_channels))

        if bias and concat:  # comment：bias要不要自己决定的啊
            self.bias = Parameter(torch.FloatTensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.negative_slope = -0.2
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_.weight)
        glorot(self.att_l)
        glorot(self.att_r)

    def forward(self,
                coords: Union[Tensor, OptPairTensor],
                features: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                bd_mask,
                poly_mesh
                ):
        self.bd_mask = bd_mask.squeeze().bool()
        self.poly_mesh = poly_mesh
        self.find_boundary(coords)
        # coords：各个节点的坐标（其实就是features的前两个纬度）
        H, C = self.heads, self.out_channels
        x_l = x_r = self.lin_l(
            features).view(-1, H, C)  # [num_node , heads, out_channels]

        x_coords_l = x_coords_r = coords  # [119, 2]

        alpha_l = (
            x_l * self.att_l).sum(dim=-1)  # [119, 6] 因为 attention
        alpha_r = (x_r * self.att_r).sum(dim=-1)  # [119, 6]

        x_coords_l = x_coords_r = coords.unsqueeze(1)  # （119, 1, 2）

        # TODO：这里的alpha_l和alpha_r为啥需要乘以个0.2？？
        out_coords = self.propagate(
            edge_index, x=(x_coords_l, x_coords_r),
            alpha=(0.2 * alpha_l, 0.2 * alpha_r))  # [119, 6, 2]

        out_coords = out_coords.mean(dim=1)  # [119, 6, 2] --> [119, 2]

        out_features = self.propagate(
            edge_index, x=(x_l, x_r),
            alpha=(alpha_l, alpha_r))  # [119, 6, 40]

        out_features = out_features.mean(dim=1)  # [119, 40]
        out_features = F.selu(out_features)  # [119, 40]  # TODO：这个可以去掉么？？
        self.fix_boundary(out_coords)

        return out_coords, out_features

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if alpha_i is None:
            alpha = alpha_j
        else:
            alpha = alpha_j + alpha_i  # comment:应该是走了这一步，因为有这两个都有的啊。。
        alpha = F.selu(alpha)
        # 这边 softmax 只要汇点信息是有原因的哦。
        alpha = softmax(alpha, index, ptr, size_i)
        # 这个函数通过广播的操作，将最后的一个纬度给扩充了。
        return x_j * alpha.unsqueeze(-1)

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

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
