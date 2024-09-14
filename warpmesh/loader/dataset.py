# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
import glob
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from cluster_utils import get_new_edges  # noqa

# from torch_geometric.loader import DataLoader as geoDataLoader

__all__ = ["MeshDataset", "MeshLoader", "MeshData", "normalise", "AggreateDataset"]


class AggreateDataset(Dataset):
    """Aggregate multiple datasets into a single dataset.

    Attributes:
        datasets (list): List of datasets.
        datasets_len (list): Length of each dataset in `datasets`.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.datasets_len = [len(dataset) for dataset in datasets]

    def __len__(self):
        """Return the total number of samples in all datasets."""
        return sum(self.datasets_len)

    def __getitem__(self, idx):
        """Fetch an individual data point from the aggregated dataset.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            tuple: The sample fetched from one of the aggregated datasets.
        """
        dataset_idx = 0
        while idx >= self.datasets_len[dataset_idx]:
            idx -= self.datasets_len[dataset_idx]
            dataset_idx += 1
        return self.datasets[dataset_idx][idx]


class MeshDataset(Dataset):
    """Dataset for mesh-based data.

    Attributes:
        x_feature (list): List of feature names for node features.
        mesh_feature (list): List of feature names for mesh features.
        conv_feature (list): List of feature names for convolution features.
        file_names (list): List of filenames containing mesh data.
    """

    def __init__(
        self,
        file_dir,
        transform=None,
        target_transform=None,
        x_feature=[
            "coord",
            "bd_mask",
            "bd_left_mask",
            "bd_right_mask",
            "bd_down_mask",
            "bd_up_mask",
        ],
        mesh_feature=[
            "coord",
            "u",
        ],
        conv_feature=[
            "conv_uh",
        ],
        conv_feature_fix=[
            "conv_uh_fix",
        ],
        load_analytical=False,
        load_jacobian=False,
        use_cluster=False,
        use_run_time_cluster=False,
        r=0.35,
        M=25,
        dist_weight=False,
        add_nei=True,
    ):
        # x feature contains the coordiate related features
        self.x_feature = x_feature
        # mesh feature is used to construct the edge realted features
        self.mesh_feature = mesh_feature
        # conv_feat, which is passed to a cnn list
        self.conv_feature = conv_feature
        # conv_feat_fix, which is passed to a cnn list
        self.conv_feature_fix = conv_feature_fix

        self.file_dir = file_dir
        file_path = os.path.join(self.file_dir, "data_*.npy")
        self.file_names = glob.glob(file_path)
        self.file_names = sorted(
            self.file_names, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        self.transform = transform
        self.target_transform = target_transform
        # if True, load the params used to generate the data
        self.load_analytical = load_analytical
        # if True, load the jacobian and jacobian det
        self.load_jacobian = load_jacobian
        # if True, use the cluster to sample the neighbors
        self.use_cluster = use_cluster
        # if True, use the run time cluster to sample the neighbors
        self.use_run_time_cluster = use_run_time_cluster
        # params for run time cluster
        self.r = r
        self.M = M
        self.dist_weight = dist_weight
        self.add_nei = add_nei
        # load phi of the MA solution

    def get_x_feature(self, data):
        """
        Extracts and concatenates the x_features for each node from the data.

        Args:
            data (dict): The data dictionary loaded from a .npy file.

        Returns:
            tensor: The concatenated x_features for each node.
        """

        x_list = []
        for key in self.x_feature:
            feat = data.item().get(key)
            if len(feat.shape) == 1:
                feat = feat.reshape(-1, 1)
            x_list.append(feat)
        x = np.concatenate(x_list, axis=1)
        x = torch.from_numpy(x).float()
        return x

    def get_mesh_feature(self, data):
        """
        Extracts and concatenates the mesh_features from the data.

        Args:
            data (dict): The data dictionary loaded from a .npy file.

        Returns:
            tensor: The concatenated mesh_features.
        """
        mesh_list = []
        for key in self.mesh_feature:
            feat = data.item().get(key)
            if len(feat.shape) == 1:
                feat = feat.reshape(-1, 1)
            mesh_list.append(feat)
        mesh = np.concatenate(mesh_list, axis=1)
        mesh = torch.from_numpy(mesh).float()
        return mesh

    def get_conv_feature(self, data):
        """
        Extracts and concatenates the conv_features from the data.

        Args:
            data (dict): The data dictionary loaded from a .npy file.

        Returns:
            tensor: The concatenated conv_features.
        """
        conv_list = []
        for key in self.conv_feature:
            feat = data.item().get(key)
            conv_list.append(feat)
        conv = np.concatenate(conv_list, axis=0)
        conv = torch.from_numpy(conv).float()
        return conv

    def get_conv_feature_fix(self, data):
        """
        Extracts and concatenates the conv_features from the data.

        Args:
            data (dict): The data dictionary loaded from a .npy file.

        Returns:
            tensor: The concatenated conv_features.
        """
        conv_list = []
        for key in self.conv_feature_fix:
            feat = data.item().get(key)
            conv_list.append(feat)
        conv = np.concatenate(conv_list, axis=0)
        conv = torch.from_numpy(conv).float()
        return conv

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Loads and returns a mesh data sample and its target from a .npy file.

        Args:
            idx (int): The index of the .npy file to load.

        Returns:
            MeshData: A MeshData object containing the sample and target.
        """
        data_path = self.file_names[idx]
        data = np.load(data_path, allow_pickle=True)
        num_nodes = torch.tensor([data.item().get("x").shape[0]])

        # advance version
        train_data = MeshData(
            x=self.get_x_feature(data),  # noqa: x here is the coordinate related features
            bd_mask=torch.from_numpy(data.item().get("bd_mask")).int(),
            conv_feat=self.get_conv_feature(data),
            # conv_feat_fix=self.get_conv_feature_fix(data),
            conv_feat_fix=self.get_conv_feature(data),
            mesh_feat=self.get_mesh_feature(data),
            edge_index=torch.from_numpy(data.item().get("edge_index_bi")).to(
                torch.int64
            ),
            y=torch.from_numpy(data.item().get("y")).float(),
            face=(
                torch.from_numpy(data.item().get("face_idxs")).to(torch.long).T
                if data.item().get("face_idxs") is not None
                else None
            ),  # noqa: E501
            phi=(
                torch.from_numpy(data.item().get("phi")).float()
                if data.item().get("phi") is not None
                else None
            ),  # noqa: E501
            grad_phi=(
                torch.from_numpy(data.item().get("grad_phi")).float()
                if data.item().get("grad_phi") is not None
                else None
            ),  # noqa: E501
            f=(
                torch.from_numpy(data.item().get("f")).float()
                if data.item().get("f") is not None
                else None
            ),  # noqa
            monitor_val=(
                torch.from_numpy(data.item().get("monitor_val")).float()
                if data.item().get("monitor_val") is not None
                else None
            ),  # noqa: E501
            node_num=num_nodes,
            poly_mesh=(
                data.item().get("poly_mesh")
                if data.item().get("poly_mesh") is not None
                else False
            ),  # noqa: E501
        )

        if self.load_analytical:
            train_data.dist_params = {
                "σ_x": data.item().get("σ_x"),
                "σ_y": data.item().get("σ_y"),
                "μ_x": data.item().get("μ_x"),
                "μ_y": data.item().get("μ_y"),
                "z": data.item().get("z"),
                "w": data.item().get("w"),
                "simple_u": data.item().get("use_iso"),
                "n_dist": data.item().get("n_dist"),
            }

        if self.load_jacobian:
            train_data.jacobian = torch.from_numpy(data.item().get("jacobian"))
            train_data.jacobian_det = torch.from_numpy(data.item().get("jacobian_det"))

        if self.transform:
            train_data = self.transform(train_data)
        if self.use_cluster:
            train_data.edge_index = data.item().get("cluster_edges").to(torch.int64)  # noqa
        if self.use_run_time_cluster:
            train_data.edge_index = get_new_edges(
                num_nodes,
                train_data.x[:, :2],
                train_data.edge_index,
                r=self.r,
                M=self.M,
                dist_weight=self.dist_weight,
                add_nei=self.add_nei,
            )
        return train_data


class MeshData(Data):
    """
    Custom PyTorch Data object designed to handle mesh data features.P

    This class is intended to be used as the base class of data samples
    returned by the MeshDataset.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        # conv_feat is feeded into cnn, so another dim is needed
        if key == "conv_feat":
            return None
        if key == "conv_feat_fix":
            return None
        if key == "node_num":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


def MeshLoader(dataset, batch_size=10, shuffle=True):
    def collate_fn(batch):
        return [item for item in batch]

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def normalise(data):
    """
    Normalizes the mesh and convolution features of a given MeshData object.

    Args:
        data (MeshData): The MeshData object containing features to normalize.

    Returns:
        MeshData: The MeshData object with normalized features.
    """
    # normalise mesh feature (only last dims, first 2 dim is coordinate)
    # Compute minimum and maximum values along the second axis
    mesh_val_feat = data.mesh_feat[:, 2:]  # value feature (no coord)

    min_val = torch.min(mesh_val_feat, dim=0).values
    max_val = torch.max(mesh_val_feat, dim=0).values
    max_abs_val = torch.max(torch.abs(min_val), torch.abs(max_val))
    data.mesh_feat[:, 2:] = data.mesh_feat[:, 2:] / max_abs_val

    # normalise conv feature
    # that is, uh and hessian norm
    conv_feat_shape = data.conv_feat.shape
    conv_feat = data.conv_feat
    conv_feat = conv_feat.reshape(conv_feat_shape[0], -1)
    min_val = torch.min(conv_feat, dim=1).values
    max_val = torch.max(conv_feat, dim=1).values
    max_abs_val = torch.max(torch.abs(min_val), torch.abs(max_val))
    max_abs_val = max_abs_val.reshape(-1, 1)
    conv_feat[:, :] = conv_feat[:, :] / max_abs_val[:, :]
    data.conv_feat = conv_feat.reshape(conv_feat_shape)

    # normalise conv_fix feature
    conv_feat_fix_shape = data.conv_feat_fix.shape
    conv_feat_fix = data.conv_feat_fix
    conv_feat_fix = conv_feat_fix.reshape(conv_feat_fix_shape[0], -1)
    min_val = torch.min(conv_feat_fix, dim=1).values
    max_val = torch.max(conv_feat_fix, dim=1).values
    max_abs_val = torch.max(torch.abs(min_val), torch.abs(max_val))
    max_abs_val = max_abs_val.reshape(-1, 1)
    conv_feat_fix[:, :] = conv_feat_fix[:, :] / max_abs_val[:, :]
    data.conv_feat_fix = conv_feat_fix.reshape(conv_feat_fix_shape)

    return data
