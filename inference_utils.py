import os

import firedrake as fd
import numpy as np
import torch
from firedrake.cython.dmcommon import facet_closure_nodes

import warpmesh as wm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_conv_feat(mesh, monitor_val, fix_reso_x=20, fix_reso_y=20):
    """
    Generate features for convolution. This involves grid spacing and other
    related features.
    """
    # if poly_mesh:
    #     return get_conv_feat_poly()
    coords = mesh.coordinates.dat.data_ro
    x_start, y_start = np.min(coords, axis=0)
    x_end, y_end = np.max(coords, axis=0)
    # fix resolution sampling (sample at fixed grid)
    conv_x_fix = np.linspace(x_start, x_end, fix_reso_x)
    conv_y_fix = np.linspace(y_start, y_end, fix_reso_y)
    conv_monitor_val_fix = np.zeros((1, len(conv_x_fix), len(conv_y_fix)))
    for i in range(len(conv_x_fix)):
        for j in range(len(conv_y_fix)):
            # (x, y) conv_feat
            try:
                conv_monitor_val_fix[:, i, j] = monitor_val.at(
                    [conv_x_fix[i], conv_y_fix[j]], tolerance=1e-3
                )
            except fd.function.PointNotInDomainError:
                conv_monitor_val_fix[:, i, j] = 0.0

    conv_monitor_val = conv_monitor_val_fix
    res = np.concatenate(
        [
            conv_monitor_val,
        ],
        axis=0,
    )
    return res


def find_edges(mesh, function_space):
    """
    Find the edges of the mesh and update the 'edges' attribute.
    """
    mesh_node_count = mesh.coordinates.dat.data_ro.shape[0]
    cell_node_list = function_space.cell_node_list
    faces = torch.from_numpy(cell_node_list)
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)
    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)
    edges_hash = mesh_node_count * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

    edges_packed = torch.stack(
        [torch.div(u, mesh_node_count, rounding_mode="floor"), u % mesh_node_count],
        dim=1,
    )

    edges_packed_reverse = edges_packed.clone()[:, [1, 0]]
    edge_bi = torch.cat([edges_packed, edges_packed_reverse], dim=0)

    edge_bi_T = edge_bi.T.numpy()
    return edge_bi_T


def find_bd(mesh, function_space, use_4_edge=False, poly_mesh=False):
    """
    Identify the boundary nodes of the mesh and update various boundary
        masks.
    """
    x_start = y_start = 0
    x_end = y_end = 1
    num_all_nodes = len(mesh.coordinates.dat.data_ro)
    coordinates = mesh.coordinates.dat.data_ro
    # boundary nodes solved by firedrake
    bd_idx = facet_closure_nodes(function_space, "on_boundary")

    # create mask for boundary nodes
    bd_mask = np.zeros(num_all_nodes).astype(bool)
    bd_mask[bd_idx] = True

    left_bd = None
    right_bd = None
    down_bd = None
    up_bd = None

    # boundary nodes solved using location of nodes
    if not poly_mesh and use_4_edge:
        left_bd = ((coordinates[:, 0] == x_start).astype(int).reshape(-1, 1),)  # noqa
        right_bd = ((coordinates[:, 0] == x_end).astype(int).reshape(-1, 1),)  # noqa
        down_bd = ((coordinates[:, 1] == y_start).astype(int).reshape(-1, 1),)  # noqa
        up_bd = ((coordinates[:, 1] == y_end).astype(int).reshape(-1, 1),)  # noqa
        left_bd = left_bd[0]
        right_bd = right_bd[0]
        down_bd = down_bd[0]
        up_bd = up_bd[0]

    return bd_mask, left_bd, right_bd, down_bd, up_bd


class InputPack:
    def __init__(
        self,
        coord,
        monitor_val,
        edge_index,
        bd_mask,
        conv_feat,
        poly_mesh=False,
        stack_boundary=True,
    ) -> None:
        self.coord = torch.tensor(coord).float().to(device)
        self.conv_feat = torch.tensor(conv_feat).float().to(device)
        self.mesh_feat = (
            torch.concat([torch.tensor(coord), torch.tensor(monitor_val)], dim=1)
            .float()
            .to(device)
        )
        self.edge_index = torch.tensor(edge_index).to(torch.int64).to(device)
        self.bd_mask = torch.tensor(bd_mask).reshape(-1, 1).to(device)
        self.node_num = torch.tensor(self.coord.shape[0]).to(device)
        self.poly_mesh = poly_mesh
        if stack_boundary:
            self.x = torch.concat(
                [
                    self.coord,
                    self.bd_mask,
                    self.bd_mask,
                    self.bd_mask,
                    self.bd_mask,
                    self.bd_mask,
                ],
                dim=1,
            ).to(device)
        else:
            self.x = torch.concat([self.coord, self.bd_mask], dim=1).to(device)

    def __repr__(self) -> str:
        return f"coord: {self.coord.shape}, conv_feat: {self.conv_feat.shape}, mesh_feat: {self.mesh_feat.shape}, edge_index: {self.edge_index.shape}, bd_mask: {self.bd_mask.shape}, node_num: {self.node_num}"


def load_model(run, config, epoch, experiment_dir):
    """
    Load Model to evaluate, prepare datasets to use.
    Also Make dir for evaluation. All evaluation files will be stored
    under the dir created.

    Args:
        config (SimpleNamespace): config for the model run
        ds_root (str): path to root data folder.
        epoch: number of epoch the model been loaded from.

    Returns:
        model: the model loaded from wandb api.
        dataset: the dataset used to train the model.
        eval_dir: the path of root dir of evaluation files.
    """

    target_file_name = "model_{}.pth".format(epoch)
    model_file = None
    for file in run.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(root=experiment_dir, replace=True)
            target_file_name = file.name
    assert model_file is not None, "Model file not found"
    model = None
    if config.model_used == "M2N":
        model = wm.M2N(
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
            deform_in_c=config.num_deform_in,
        )
    elif config.model_used == "MRN":
        model = wm.MRN(
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
            deform_in_c=config.num_deform_in,
            num_loop=config.num_deformer_loop,
        )
    elif config.model_used == "MRT" or config.model_used == "MRTransformer":
        model = wm.MRTransformer(
            num_transformer_in=config.num_transformer_in,
            num_transformer_out=config.num_transformer_out,
            num_transformer_embed_dim=config.num_transformer_embed_dim,
            num_transformer_heads=config.num_transformer_heads,
            num_transformer_layers=config.num_transformer_layers,
            transformer_training_mask=config.transformer_training_mask,
            transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,  # noqa
            transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,  # noqa
            deform_in_c=config.num_deform_in,
            deform_out_type=config.deform_out_type,
            num_loop=config.num_deformer_loop,
            device=device,
        )
    elif config.model_used == "M2T":
        model = wm.M2T(
            num_transformer_in=config.num_transformer_in,
            num_transformer_out=config.num_transformer_out,
            num_transformer_embed_dim=config.num_transformer_embed_dim,
            num_transformer_heads=config.num_transformer_heads,
            num_transformer_layers=config.num_transformer_layers,
            transformer_training_mask=config.transformer_training_mask,
            transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,  # noqa
            transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,  # noqa
            deform_in_c=config.num_deform_in,
            local_feature_dim_in=config.num_lfe_in,
            deform_out_type=config.deform_out_type,
            num_loop=config.num_deformer_loop,
            device=device,
        )
    elif config.model_used == "M2N_T":
        model = wm.M2N_T(
            deform_in_c=config.num_deform_in,
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
        )
    else:
        print("Model not found")
    model_file_path = os.path.join(experiment_dir, target_file_name)
    model = wm.load_model(model, model_file_path)
    return model
