# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os
import numpy as np
import torch
from torch_geometric.data import Data
from firedrake.cython.dmcommon import facet_closure_nodes

os.environ['OMP_NUM_THREADS'] = "1"
__all__ = ["MeshProcessor"]


class MeshProcessor():
    """
    MeshProcessor class for pre-processing mesh data, attaching features to
        nodes,
    and converting them to training data.

    Parameters:
    - original_mesh: The initial mesh.
    - optimal_mesh: The optimal mesh after adaptation.
    - function_space: The function space over which the mesh is defined.
    - use_4_edge: Whether to use four edges for finding boundaries.
    - feature: Dictionary containing features like 'uh', 'grad_uh' etc.
    - raw_feature: Dictionary containing raw features like 'uh', 'hessian_norm
        etc.
    - dist_params: Dictionary containing distribution parameters.

    Attributes:
    - dist_params: Distribution parameters.
    - mesh: The original mesh.
    - optimal_mesh: The optimal mesh.
    - function_space: The function space.
    - feature: The attached features.
    - raw_feature: The raw features.
    - coordinates: The coordinates of the original mesh.
    - optimal_coordinates: The coordinates of the optimal mesh.
    - cell_node_list: The list of nodes for each cell.
    - num_nodes: The number of nodes in each cell.
    """
    def __init__(
            self, original_mesh, optimal_mesh,
            function_space, use_4_edge=True,
            feature={
                "uh": None,
                "grad_uh": None,
            },
            raw_feature={
                "uh": None,
                "hessian_norm": None,
            },
            dist_params={
                "n_dist": None,
                "σ_x": None,
                "σ_y": None,
                "μ_x": None,
                "μ_y": None,
                "z": None,
                "w": None,
                "use_iso": None,
            }
    ):
        self.dist_params = dist_params
        self.mesh = original_mesh
        self.optimal_mesh = optimal_mesh
        # the optimal mesh function space
        self.function_space = function_space
        self.feature = feature
        self.raw_feature = raw_feature
        self.coordinates = self.mesh.coordinates.dat.data_ro  # (num_nodes, 2)
        self.x = self.coordinates
        self.optimal_coordinates = self.optimal_mesh.coordinates.dat.data_ro  # noqa (num_nodes, 2) 
        self.y = self.optimal_coordinates  # (num_nodes, 2), ground truth
        self.cell_node_list = self.function_space.cell_node_list
        self.num_nodes = self.cell_node_list.shape[1]
        self.find_edges()
        self.find_bd()
        self.attach_feature()
        self.conv_feat = self.get_conv_feat()
        self.to_train_data()

    def get_conv_feat(self, fix_reso_x=20, fix_reso_y=20):
        """
        Generate features for convolution. This involves grid spacing and other
        related features.
        """
        coords = self.mesh.coordinates.dat.data_ro
        x_start, y_start = np.min(coords, axis=0)
        x_end, y_end = np.max(coords, axis=0)

        # fix resolution sampling (sample at fixed grid)
        conv_x_fix = np.linspace(x_start, x_end, fix_reso_x)
        conv_y_fix = np.linspace(y_start, y_end, fix_reso_y)
        conv_xy_fix = np.zeros((2, fix_reso_x, fix_reso_y))
        conv_uh_fix = np.zeros((1, len(conv_x_fix), len(conv_y_fix)))
        conv_hessian_norm_fix = np.zeros((1, len(conv_x_fix), len(conv_y_fix)))
        for i in range(len(conv_x_fix)):
            for j in range(len(conv_y_fix)):
                # (x, y) conv_feat
                conv_xy_fix[:, i, j] = np.array([conv_x_fix[i], conv_y_fix[j]])
                conv_uh_fix[:, i, j] = self.raw_feature["uh"].at(
                    [conv_x_fix[i],
                     conv_y_fix[j]])
                conv_hessian_norm_fix[:, i, j] = self.raw_feature[
                    "hessian_norm"].at(
                    [conv_x_fix[i],
                     conv_y_fix[j]])
        self.conv_xy_fix = conv_xy_fix
        self.conv_uh_fix = conv_uh_fix
        self.conv_hessian_norm_fix = conv_hessian_norm_fix

        # dynamic resolution sampling (sampled at mesh nodes)
        x_coords_unique = np.unique(coords[:, 0])
        y_coords_unique = np.unique(coords[:, 1])
        conv_x = np.linspace(x_start, x_end, len(x_coords_unique))
        conv_y = np.linspace(y_start, y_end, len(y_coords_unique))

        conv_uh = np.zeros((1, len(conv_x), len(conv_y)))
        conv_hessian_norm = np.zeros((1, len(conv_x), len(conv_y)))
        conv_xy = np.zeros((2, len(conv_x), len(conv_y)))
        for i in range(len(conv_x)):
            for j in range(len(conv_y)):
                # (x, y) conv_feat
                conv_xy[:, i, j] = np.array([conv_x[i], conv_y[j]])
                # uh conv_feat
                conv_uh[:, i, j] = self.raw_feature["uh"].at(
                    [conv_x[i],
                     conv_y[j]],
                    tolerance=1e-4)
                # uh norm conv_feat
                conv_hessian_norm[:, i, j] = self.raw_feature[
                    "hessian_norm"].at(
                    [conv_x[i],
                     conv_y[j]],
                    tolerance=1e-4)
        self.conv_xy = conv_xy
        self.conv_uh = conv_uh
        self.conv_hessian_norm = conv_hessian_norm
        res = np.concatenate(
            [conv_xy, conv_uh, conv_hessian_norm], axis=0)
        return res

    def attach_feature(self):
        """
        Attach features to nodes of the mesh. The features to be attached are
            specified
        in the 'feature' attribute.
        """
        for key in self.feature:
            if (self.feature[key] is not None):
                self.x = np.concatenate([self.x, self.feature[key]], axis=1)
        return

    def to_train_data(self):
        """
        Convert mesh and associated features to PyTorch Geometric Data format.
        This can be used directly for machine learning training.
        """

        x_tensor = torch.tensor(self.x, dtype=torch.float)
        y_tensor = torch.tensor(self.y, dtype=torch.float)
        edge_index_tensor = torch.tensor(self.edges, dtype=torch.long)
        scale = self.mesh.coordinates.dat.data_ro.max(axis=0) - \
            self.mesh.coordinates.dat.data_ro.min(axis=0)

        data = Data(x=x_tensor,
                    edge_index=edge_index_tensor,
                    y=y_tensor,
                    pos=self.coordinates
                    )
        data.scale = torch.tensor(scale, dtype=torch.float)
        data.cell_node_list = self.cell_node_list
        data.conv_feat = torch.from_numpy(self.conv_feat).float()

        np_data = {
            "x": self.x,
            "coord": self.coordinates,
            "u": self.feature["uh"],
            "grad_u": self.feature["grad_uh"],
            "hessian": self.feature["hessian"],
            "hessian_norm": self.feature["hessian_norm"],
            "jacobian": self.feature["jacobian"],
            "jacobian_det": self.feature["jacobian_det"],
            "edge_index": self.edges,
            "y": self.y,
            "pos": self.coordinates,
            "scale": scale,
            "cell_node_list": self.cell_node_list,
            "conv_feat": self.conv_feat,
            "bd_mask": self.bd_mask.astype(int).reshape(-1, 1),
            "bd_left_mask": self.left_bd.astype(int).reshape(-1, 1),
            "bd_right_mask": self.right_bd.astype(int).reshape(-1, 1),
            "bd_down_mask": self.down_bd.astype(int).reshape(-1, 1),
            "bd_up_mask": self.up_bd.astype(int).reshape(-1, 1),
            "conv_xy": self.conv_xy,
            "conv_uh": self.conv_uh,
            "conv_hessian_norm": self.conv_hessian_norm,
            "conv_xy_fix": self.conv_xy_fix,
            "conv_uh_fix": self.conv_uh_fix,
            "conv_hessian_norm_fix": self.conv_hessian_norm_fix,
            "σ_x": self.dist_params["σ_x"],
            "σ_y": self.dist_params["σ_y"],
            "μ_x": self.dist_params["μ_x"],
            "μ_y": self.dist_params["μ_y"],
            "z": self.dist_params["z"],
            "w": self.dist_params["w"],
            "use_iso": self.dist_params["use_iso"],
            "n_dist": self.dist_params["n_dist"],
        }
        print("data saved, details:")
        print("conv_feat shape: ", self.conv_feat.shape)
        print("x shape: ", self.x.shape)

        self.np_data = np_data
        self.train_data = data
        return data

    def find_edges(self):
        """
        Find the edges of the mesh and update the 'edges' attribute.
        """
        edges = []
        cell_node_list = self.cell_node_list.T
        cell_node_list = np.concatenate(
            [cell_node_list, cell_node_list[0:1, :]])
        for i in range(self.num_nodes):
            edges_temp = np.concatenate(
                [cell_node_list[i:i+1, :], cell_node_list[i+1:i+2, :]],
                axis=0
            ).T
            edges.append(edges_temp)
        edges = np.concatenate(edges, axis=0)
        edges = np.unique(edges, axis=0).T  # make it pytorch-geo friendly
        self.edges = edges
        return edges

    def find_bd(self):
        """
        Identify the boundary nodes of the mesh and update various boundary
            masks.
        """
        use_4_edge = True
        x_start = y_start = 0
        x_end = y_end = 1
        num_all_nodes = len(self.mesh.coordinates.dat.data_ro)
        # boundary nodes solved by firedrake
        self.bd_idx = facet_closure_nodes(
            self.function_space, "on_boundary")

        # create mask for boundary nodes
        self.bd_mask = np.zeros(num_all_nodes).astype(bool)
        self.bd_mask[self.bd_idx] = True

        # boundary nodes solved using location of nodes
        if (use_4_edge):
            self.left_bd = (self.coordinates[:, 0] == x_start)
            self.right_bd = (self.coordinates[:, 0] == x_end)
            self.down_bd = (self.coordinates[:, 1] == y_start)
            self.up_bd = (self.coordinates[:, 1] == y_end)
            self.bd_all = np.any(
                [self.left_bd, self.right_bd, self.down_bd, self.up_bd],
                axis=0
            )
        return

    def save_taining_data(self, path):
        """
        Save the processed data into disk for future use.

        Parameters:
        - path: The directory where to save the data.
        """
        # torch.save(self.train_data, path + ".train")
        np.save(path + ".npy", self.np_data)
        return
