import os
import warpmesh as wm
import torch
import warnings
from torch_geometric.data import DataLoader

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

project_dir = os.path.dirname(
    os.path.dirname((os.path.abspath(__file__))))
data_set_path = (
    "/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/"
    "z=<0,1>_ndist=None_max_dist=6_<20x20>_n=400_cmplx"
)
data_path = data_set_path

data_set = wm.MeshDataset(
    os.path.join(data_path, "train"),
    transform=wm.normalise,
)

loader = DataLoader(data_set, batch_size=10, shuffle=False)


#  =================TRAIN=======================================
if __name__ == "__main__":
    print("in play_loader.py")
    batch = next(iter(loader))
    print(batch)
    print(batch.y.shape)
