import os
import warnings

import torch
from torch_geometric.loader import DataLoader

import warpmesh as wm

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = wm.MRN(deform_in_c=3, num_loop=3)

x_feat = [
    "coord",
    "bd_mask",
    # "bd_left_mask",
    # "bd_right_mask",
    # "bd_down_mask",
    # "bd_up_mask",
]
mesh_feat = [
    "coord",
    "u",
    "hessian_norm",
]
conv_feat = [
    "conv_uh",
    "conv_hessian_norm",
]

is_normalise = True

project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
data_set_path = os.path.join(project_dir, "data/")
data_path = data_set_path

data_set = wm.MeshDataset(
    os.path.join(data_path, "test"),
    transform=wm.normalise if is_normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)

loader = DataLoader(data_set, batch_size=10, shuffle=False)
batch = next(iter(loader))

#  =================TRAIN=======================================
if __name__ == "__main__":
    data = data_set[0]
    out = model(batch)
