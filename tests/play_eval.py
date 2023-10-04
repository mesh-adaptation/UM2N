import warpmesh as wm
import torch
from torch_geometric.data import DataLoader

import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available()
else 'cpu') # noqa

batch_size = 10

loss_func = torch.nn.L1Loss()


data_dir = '/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<20x20>_n=400_smpl/val'  # noqa

weight_path = "/Users/cw1722/Downloads/model_1499 (7).pth"


prediction_dir = "/Users/cw1722/Documents/irp/irp-cw1722/data/temp"


model = wm.MRN(
    deform_in_c=7,
    gfe_in_c=2,
    lfe_in_c=4,
    num_loop=3
).to(device)

model = wm.load_model(model, weight_path)

n_elem_x = n_elem_y = 20


normalise = True

x_feat = [
    "coord",
    "bd_mask",
    "bd_left_mask",
    "bd_right_mask",
    "bd_down_mask",
    "bd_up_mask",
]
mesh_feat = [
    "coord",
    "u",
    "hessian_norm",
    # "grad_u",
    # "hessian",
]
conv_feat = [
    "conv_uh",
    "conv_hessian_norm",
]

data_set = wm.MeshDataset(
    data_dir,
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)

loader = DataLoader(data_set, batch_size=batch_size)

loss = wm.evaluate(loader, model, device, loss_func)
print(loss)
