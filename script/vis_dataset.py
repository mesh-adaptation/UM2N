# Author: Chunyang Wang
# GitHub Username: acse-cw1722

# %% setup
import firedrake as fd
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import UM2N

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

# %%
# Plot M2N dataset in a 2*2 grid
cmap = "coolwarm"
fontsize = 20
offset = 179
# offset = 39
n_grid = 20
n_row = 2
normalise = True
# data_set_type = "smpl"
data_set_type = "cmplx"
data_dir = (
    f"/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/"
    f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=400_"
    f"{data_set_type}/train"
)
print(data_dir)

data_set = UM2N.MeshDataset(
    data_dir,
    transform=UM2N.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
    load_analytical=True,
)

fig = plt.figure(figsize=(12, 10.5))
for i in range(n_row):
    for j in range(n_row):
        idx = i * n_row + j + offset
        ax = fig.add_subplot(n_row, n_row, i * n_row + j + 1)
        data = data_set[idx]
        mesh = fd.UnitSquareMesh(n_grid, n_grid)
        mesh.coordinates.dat.data[:] = data["y"][:]
        u = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        u.dat.data[:] = data["mesh_feat"][:, 2]

        # fd.trisurf(u, axes=ax, cmap=cmap)

        plot_obj = fd.tripcolor(u, axes=ax, cmap=cmap)
        cbar = plt.colorbar(plot_obj, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        fd.triplot(mesh, axes=ax)
        border = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=fig.transFigure,
            color="black",
            fill=False,
            zorder=10,
            linewidth=3,
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        fig.patches.append(border)

# %%
# Plot Unifor mesh
n_grid = 15
pure_mesh = fd.UnitSquareMesh(n_grid, n_grid)
fig, ax = plt.subplots(figsize=(10, 10))
fd.triplot(pure_mesh, axes=ax)

# %%
# plot hessian norm and solution field
cmap = "coolwarm"
fontsize = 20
idx = 17
# offset = 36
n_grid = 15
normalise = True
# data_set_type = "smpl"
data_set_type = "smpl"
data_dir = (
    f"/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/"
    f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=400_"
    f"{data_set_type}/train"
)
print(data_dir)

data_set = UM2N.MeshDataset(
    data_dir,
    transform=UM2N.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
    load_analytical=True,
)

fig, ax = plt.subplots(figsize=(10, 10))

data = data_set[idx]
mesh = fd.UnitSquareMesh(n_grid, n_grid)
mesh.coordinates.dat.data[:] = data["y"][:]
u = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
u.dat.data[:] = data["mesh_feat"][:, 2]
hessian_norm = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
hessian_norm.dat.data[:] = data["mesh_feat"][:, 3]
fd.triplot(mesh, axes=ax)

# %%
