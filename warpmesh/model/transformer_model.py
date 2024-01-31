import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat


class MLP_model(torch.nn.Module):
    def __init__(self, input_channels, output_channels, list_hiddens=[128, 128], hidden_act="LeakyReLU", output_act="LeakyReLU", 
                 input_norm=None, dropout_prob=0.0):
        """
        Note that list_hiddens should be a list of hidden channels per MLP layers
        e.g. [64 128 64]

        Args:
            input_channels (_type_): _description_
            list_hiddens (_type_): _description_
            output_channels (_type_): _description_
            hidden_act (str, optional): _description_. Defaults to "LeakyReLU".
            output_act (str, optional): _description_. Defaults to "LeakyReLU".
            dropout_prob (float, optional): _description_. Defaults to 0.0.
            input_norm (float, optional): one of ["BatchNorm1d", "LayerNorm"]
        """
        super(MLP_model, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = list_hiddens
        self.hidden_act = getattr(nn, hidden_act)()
        self.output_act = getattr(nn, output_act)()
        
        list_in_channels = [input_channels] + list_hiddens
        list_out_channels = list_hiddens + [output_channels]
        
        list_layers = []
        for i, (in_channels, out_channels) in enumerate(zip(list_in_channels, list_out_channels)):
            list_layers.append(nn.Linear(in_channels, out_channels))
            # output layer
            if i == len(list_in_channels): 
                list_layers.append(self.output_act)
            else:
                list_layers.append(self.hidden_act)
        self.layers = nn.ModuleList(list_layers)
        
        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)
        
        if input_norm is not None:
            if input_norm == "batch":
                self.input_norm = nn.BatchNorm1d(input_channels)
            elif input_norm == "layer":
                self.input_norm = nn.LayerNorm(input_channels)
            else:
                raise NotImplementedError
        
    def forward(self, x):
        if hasattr(self, "input_norm"):
            x = self.input_norm(x)
        
        for _, layer in enumerate(self.layers):
            x = layer(x)
            if hasattr(self, "dropout"):
                x = self.dropout(x)
        return x     

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dense_ratio=4, list_dropout=[0.1, 0.1, 0.1], activation="GELU") -> None:
        super(TransformerBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dense_dim = embed_dim * dense_ratio
        
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads, dropout=list_dropout[0], add_bias_kv=False, batch_first=True)
        self.post_attn_norm = nn.LayerNorm(embed_dim)
        self.post_attn_dropout = nn.Dropout(list_dropout[1])
        
        self.pre_dense_norm = nn.LayerNorm(embed_dim)
        self.dense_1 = nn.Linear(embed_dim, self.dense_dim)
        self.activation = getattr(nn, activation)()
        self.act_dropout = nn.Dropout(list_dropout[2])
        self.post_dense_norm = nn.LayerNorm(self.dense_dim)
        
        self.dense_2 = nn.Linear(self.dense_dim, embed_dim)
        
        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) 
        self.residual_weight = nn.Parameter(torch.ones(embed_dim), requires_grad=True)

    def forward(self, x, k, v, x_cls=None, key_padding_mask=None, attn_mask=None, return_attn=False):
        # In pytorch nn.MultiheadAttention, key_padding_mask True indicates ignore the corresponding key value
        # NOTE: check default True or False of key_padding_mask with nn.MultiheadAttention
        # if key_padding_mask is not None:
        #     key_padding_mask = ~key_padding_mask
        
        if x_cls is not None:
            # to be implemented later
            pass
        else:
            residual = x
            x = self.pre_attn_norm(x)
            # NOTE: Here we use batch first in nn.MultiheadAttention
            # [batch_size, num_points, embed_dim]
            
            x, attn_scores = self.attn_layer(x, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
        if self.c_attn is not None:
            num_points = x.shape[1]
            x = x.view(-1, num_points, self.num_heads, self.head_dim)
            x = torch.einsum("b n h d, h -> b n d h", x, self.c_attn)
            x = x.reshape(-1, num_points, self.embed_dim)
        
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        
        x = self.post_attn_dropout(x)
        x = x + residual
        
        residual = x
        x = self.pre_dense_norm(x)
        x = self.activation(self.dense_1(x))
        x = self.act_dropout(x)
        
        if self.post_dense_norm is not None:
            x = self.post_dense_norm(x)
            
        x = self.dense_2(x)
        x = self.post_attn_dropout(x)
        
        if self.residual_weight is not None:
            residual = torch.mul(self.residual_weight, residual)
        
        x = x + residual
        if not return_attn:
            return x
        else:
            return x, attn_scores
    

class TransformerModel(nn.Module):
    def __init__(self, *, input_dim, embed_dim, output_dim, num_heads=4, num_layers=3) -> None:
        super(TransformerModel, self).__init__()
        # save torch module kwargs - lightning ckpt too cumbersome to use
        self.kwargs = {"input_dim": input_dim, "embed_dim": embed_dim, "output_dim": output_dim, "num_heads": num_heads, "num_layers": num_layers}
        self.num_heads = num_heads

        list_attn_layers = []
        for _ in range(num_layers):
            list_attn_layers.append(TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, list_dropout=[0.1, 0.1, 0.1]))
        self.attn_layers = nn.ModuleList(list_attn_layers)

        self.mlp_in = MLP_model(input_dim, embed_dim, [embed_dim], hidden_act="GELU", output_act="GELU")
        self.mlp_out = MLP_model(embed_dim, output_dim, [embed_dim], hidden_act="GELU", output_act="GELU")

        
    def forward(self, x, k, v, key_padding_mask=None, attention_mask=None):
        x = self.mlp_in(x)
        k = self.mlp_in(k)
        v = self.mlp_in(v)
        cnt = 0
        for _, layer in enumerate(self.attn_layers):
            if cnt == 0:
                x = layer(x, k, v, key_padding_mask=key_padding_mask, attn_mask=attention_mask)
            else:
                x = layer(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attention_mask)
        x = self.mlp_out(x)
        return x
    
    def get_attention_scores(self, x, key_padding_mask=None, attn_mask=None):
        list_attn_scores = []
        x = self.mlp_in(x)
        for _, layer in enumerate(self.attn_layers):
            x, attn_scores = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, return_attn=True)
            if key_padding_mask is not None:
                mask_mat =  rearrange(key_padding_mask, 'b i -> b i 1') * rearrange(key_padding_mask, 'b j -> b 1 j')
                num_points = key_padding_mask.sum().numpy()
                attn_mat = attn_scores.detach().numpy().squeeze()[mask_mat.numpy().squeeze()].reshape(num_points, num_points)
            else:
                # The dim for torch squeeze can not be tuple with a version lower than 2.0
                attn_mat = torch.squeeze(attn_scores, dim=0).detach().cpu().numpy()
            list_attn_scores.append(attn_mat)
            
        return list_attn_scores
    
    # def save_kwargs(self, name=None):
    #     create_folder("model_kwargs")
    #     kwargs_name = "transformer_model_kwargs_" + name + ".pkl" if name is not None else "transformer_model_kwargs.pkl"
    #     dump_pickle(self.kwargs, kwargs_name)
