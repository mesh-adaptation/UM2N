import os
import warnings

import torch

import UM2N

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
data_set_path = os.path.join(project_dir, "data/")
data_path = data_set_path

data_set = UM2N.MeshDataset(
    os.path.join(data_path, "test"),
    transform=UM2N.normalise,
)


#  =================TRAIN=======================================
if __name__ == "__main__":
    data = data_set[2]
    print("data.x", data.x.shape)
    print("data.y", data.edge_index.shape)
    print("mesh_feat", data.mesh_feat.shape)
    print("conv_feat", data.conv_feat.shape)
    print(
        "conv_feat min max after",
        torch.min(data.conv_feat[:, :2]),
        torch.max(data.conv_feat[:, :2]),
    )
    print(
        "mesh_feat min max after", torch.min(data.mesh_feat), torch.max(data.mesh_feat)
    )
    print("mesh_feat", data.mesh_feat[:200, :])
