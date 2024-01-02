# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os
import numpy as np
import torch
import firedrake as fd
from firedrake.cython.dmcommon import facet_closure_nodes

os.environ['OMP_NUM_THREADS'] = "1"
__all__ = ["MeshProcessor"]


def judge_in_hull(hull_points: np.array, point_to_judge: np.array, scale=1.0):
    # print("in judge_in_hull")
    # print("hull_points: ", hull_points)
    mean = hull_points.mean(axis=0)
    # print("mean: ", mean)
    hull_points = hull_points - (1 - scale) * (hull_points - mean)
    for i in range(len(hull_points)):
        if i == len(hull_points) - 1:
            # edge_vector = (hull_points[0, :] - hull_points[i, :])
            edge_vector = (hull_points[i, :] - hull_points[0, :])
        else:
            # edge_vector = (hull_points[i + 1, :] - hull_points[i, :])
            edge_vector = (hull_points[i, :] - hull_points[i + 1, :])
        if np.cross(edge_vector, point_to_judge - hull_points[i, :]) > 0:
            return False
    return True


def get_sample_points_in_convex_hull(points: np.array, size=32, scale=0.95):
    print("in get_sample_points_in_convex_hull")
    x_left, x_right = points[:, 0].min(), points[:, 0].max()
    y_low, y_up = points[:, 1].min(), points[:, 1].max()
    print("x_left, x_right, y_low, y_up: ", x_left, x_right, y_low, y_up)
    points_x = np.linspace(x_left, x_right, size)
    points_y = np.linspace(y_low, y_up, size)[::-1]
    coord_ = np.random.random(size=(size ** 2, 2))
    fetch_flag = np.zeros(size ** 2, dtype=bool)
    for i in range(size):
        for j in range(size):
            idx = i*size + j
            coord_[idx, 0] = points_x[j]
            coord_[idx, 1] = points_y[i]
            # print(judge_in_hull(points, coord_[idx, :], scale=scale))
            if judge_in_hull(points, coord_[idx, :], scale=scale):
                fetch_flag[idx] = 1
    fetch_coord = coord_[fetch_flag, :]
    return fetch_flag, fetch_coord


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
            poly_mesh=False, num_boundary=4,
            nu=None,
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
            },
            gauss_list=None,   # used in burgers equation bumps generation
    ):
        self.use_4_edge = use_4_edge
        self.num_boundary = num_boundary
        self.poly_mesh = poly_mesh
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
        # PDE params
        self.nu = nu
        self.gauss_list = gauss_list
        self.to_train_data()

    def get_conv_feat_poly(self):
        # print("poly poly my little poly")
        num_edges = self.num_boundary
        corner_idx_list = []
        boundary_idx_list = []
        fun_space = fd.FunctionSpace(self.mesh, "CG", 1)
        for i in range(num_edges):
            boundary_idx_list.append(
                facet_closure_nodes(fun_space, [i+1]))
        for i in range(num_edges):
            if i == num_edges-1:
                corner_idx_list.append(
                    np.intersect1d(
                        boundary_idx_list[i],
                        boundary_idx_list[0]))
                break
            corner_idx_list.append(
                np.intersect1d(
                    boundary_idx_list[i],
                    boundary_idx_list[i+1]))
        corner_idx = np.hstack(corner_idx_list)
        corner_coordinates = self.coordinates[:][corner_idx, :]
        sample_flag, sample_coord = get_sample_points_in_convex_hull(
            corner_coordinates)
        # sampling for uh
        uh_in_polygon = self.raw_feature["uh"].at(
            sample_coord, tolerance=1e-4)
        uh_sample_buffer = np.zeros((32**2, 1))
        uh_sample_buffer[sample_flag, :] = np.vstack(uh_in_polygon)
        uh_sample_buffer = uh_sample_buffer.reshape(32, 32)
        # sampling for hessian norm
        hessian_in_polygon = self.raw_feature["hessian_norm"].at(
            sample_coord, tolerance=1e-4)
        hessian_sample_buffer = np.zeros((32**2, 1))
        hessian_sample_buffer[sample_flag, :] = np.vstack(hessian_in_polygon)
        hessian_sample_buffer = hessian_sample_buffer.reshape(32, 32)
        # On poly mesh, we define conv_fix and conv as the same
        self.conv_uh = uh_sample_buffer[np.newaxis, :, :]
        self.conv_uh_fix = uh_sample_buffer[np.newaxis, :, :]
        self.conv_hessian_norm = hessian_sample_buffer[np.newaxis, :, :]
        self.conv_hessian_norm_fix = hessian_sample_buffer[np.newaxis, :, :]
        self.conv_xy = None
        self.conv_xy_fix = None
        return

    def get_conv_feat(self, fix_reso_x=20, fix_reso_y=20):
        """
        Generate features for convolution. This involves grid spacing and other
        related features.
        """
        if self.poly_mesh:
            return self.get_conv_feat_poly()
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
                     conv_y_fix[j]],
                    tolerance=1e-3
                )
                conv_hessian_norm_fix[:, i, j] = self.raw_feature[
                    "hessian_norm"].at(
                    [conv_x_fix[i],
                     conv_y_fix[j]],
                    tolerance=1e-3
                )
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
        scale = self.mesh.coordinates.dat.data_ro.max

        np_data = {
            "x": self.x,
            "coord": self.coordinates,
            "u": self.feature["uh"],
            "grad_u": self.feature["grad_uh"],
            "hessian": self.feature["hessian"],
            "phi": self.feature["phi"],
            "grad_phi": self.feature["grad_phi"],
            "hessian_norm": self.feature["hessian_norm"],
            "jacobian": self.feature["jacobian"],
            "jacobian_det": self.feature["jacobian_det"],
            "edge_index": self.edge_T,
            "edge_index_bi": self.edge_bi_T,
            "cluster_edges": None, # this will be added if we use data_transform.py to add cluster edges  # noqa
            "y": self.y,
            "pos": self.coordinates,
            "scale": scale,
            "cell_node_list": self.cell_node_list,
            "conv_feat": self.conv_feat,
            "bd_mask": self.bd_mask.astype(int).reshape(-1, 1),
            "bd_left_mask": self.left_bd,
            "bd_right_mask": self.right_bd,
            "bd_down_mask": self.down_bd,
            "bd_up_mask": self.up_bd,
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
            "face_idxs": self.cell_node_list,
            "n_dist": self.dist_params["n_dist"],
            "nu": self.nu,
            "gauss_list": self.gauss_list,
        }
        print("data saved, details:")
        # print("conv_feat shape: ", self.conv_feat.shape)
        print("x shape: ", self.x.shape)

        self.np_data = np_data
        return

    def find_edges(self):
        """
        Find the edges of the mesh and update the 'edges' attribute.
        """
        mesh_node_count = self.coordinates.shape[0]
        faces = torch.from_numpy(self.cell_node_list)
        v0, v1, v2 = faces.chunk(3, dim=1)
        e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
        e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
        e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)
        edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
        edges, _ = edges.sort(dim=1)
        edges_hash = mesh_node_count * edges[:, 0] + edges[:, 1]
        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

        edges_packed = torch.stack(
            [
                torch.div(u, mesh_node_count, rounding_mode="floor"),
                u % mesh_node_count
            ], dim=1)

        self.single_edges = edges_packed
        edges_packed_reverse = edges_packed.clone()[:, [1, 0]]
        self.edge_bi = torch.cat(
            [edges_packed,
             edges_packed_reverse], dim=0)

        self.edge_T = self.single_edges.T.numpy()
        self.edge_bi_T = self.edge_bi.T.numpy()
        return self.edge_bi_T

    def find_bd(self):
        """
        Identify the boundary nodes of the mesh and update various boundary
            masks.
        """
        use_4_edge = self.use_4_edge
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
            self.left_bd = (self.coordinates[:, 0] == x_start).astype(int).reshape(-1, 1),  # noqa
            self.right_bd = (self.coordinates[:, 0] == x_end).astype(int).reshape(-1, 1),  # noqa
            self.down_bd = (self.coordinates[:, 1] == y_start).astype(int).reshape(-1, 1),  # noqa
            self.up_bd = (self.coordinates[:, 1] == y_end).astype(int).reshape(-1, 1),  # noqa
            self.bd_all = np.any(
                [self.left_bd, self.right_bd, self.down_bd, self.up_bd],
                axis=0
            )
            self.left_bd = self.left_bd[0]
            self.right_bd = self.right_bd[0]
            self.down_bd = self.down_bd[0]
            self.up_bd = self.up_bd[0]
        # using poly mesh, set 4 edges to None
        if (self.poly_mesh):
            self.left_bd = None
            self.right_bd = None
            self.down_bd = None
            self.up_bd = None
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
