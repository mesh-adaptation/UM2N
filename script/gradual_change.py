# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# %% importimport warnings
import warpmesh as wm
import torch
import pandas as pd  # noqa
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
import firedrake as fd
import warnings
from torch_geometric.data import DataLoader  # noqa

torch.no_grad()
warnings.filterwarnings("ignore")  # noqa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% load model
n_elem = 20
data_dir = f"/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{n_elem}x{n_elem}>_n=400_smpl/val"  # noqa

M2N_weight_path = "/Users/cw1722/Downloads/M2N__15,20__cmplx/weight/model_999.pth"  # noqa
# MRN_path = "/Users/cw1722/Downloads/MRN_r=5_15,20__smpl/weight/model_999.pth"  # noqa
MRN_path = "/Users/cw1722/Downloads/MRN_r=5__15,20__cmplx/weight/model_999.pth"  # noqa

model_M2N = wm.M2N(
    deform_in_c=7,
    gfe_in_c=2,
    lfe_in_c=4,
).to(device)
model_M2N = wm.load_model(model_M2N, M2N_weight_path)

model_MRN = wm.MRN(
    deform_in_c=7,
    gfe_in_c=2,
    lfe_in_c=4,
    num_loop=5,
).to(device)
model_MRN = wm.load_model(model_MRN, MRN_path)

# %% dataset load
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
normalise = True
loss_func = torch.nn.L1Loss()
data_set = wm.MeshDataset(
    data_dir,
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
    load_analytical=True,
)
# %% gradual change experiment


def plot_gradual_change(idx, data_set=data_set, model=model_MRN):
    total_iter = 5
    data = data_set[idx]

    img_size = 10

    fig, axs = plt.subplots(1, total_iter, figsize=(img_size * total_iter, img_size))

    for i in range(total_iter):
        print(i)
        out = model.move(data, i + 1)
        loss = 1000 * loss_func(out, data.y).item()
        out = out.detach().cpu().numpy()
        mesh = fd.UnitSquareMesh(n_elem, n_elem)
        mesh.coordinates.dat.data[:] = out[:]
        fd.triplot(mesh, axes=axs[i])
        axs[i].set_title(f"n={i}, Loss: {loss:.2f}", fontsize=30)
    return fig, axs


# %%


samples = [1, 21, 33]

for i in samples:
    fig, axs = plot_gradual_change(i, data_set=data_set, model=model_MRN)
# %% on different datasets

n_nodes = [18, 19, 20, 21, 22]
total_iter = 5

res = np.zeros((len(n_nodes), total_iter))

dataset_list = [
    f"/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{n_node}x{n_node}>_n=40_cmplx/train"
    for n_node in n_nodes  # noqa
]

data_sets = [
    wm.MeshDataset(
        data_dir,
        transform=wm.normalise if normalise else None,
        x_feature=x_feat,
        mesh_feature=mesh_feat,
        conv_feature=conv_feat,
        load_analytical=True,
    )
    for data_dir in dataset_list
]

for i in range(len(n_nodes)):
    ds = data_sets[i]
    loss_arr = np.zeros((1, total_iter))
    for j in range(len(ds)):
        data = ds[j]
        for k in range(total_iter):
            model_MRN.eval()
            out = model_MRN.move(data, k + 1)
            loss = 1000 * (loss_func(out, data.y).item())
            loss_arr[0, k] += loss
        loss_arr = loss_arr / len(ds)
    res[i, :] = loss_arr

# %%
