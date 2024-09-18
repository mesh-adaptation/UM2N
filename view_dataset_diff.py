import os
from datetime import datetime

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import DataLoader

import warpmesh as wm
from warpmesh.helper import load_yaml_to_namespace
from warpmesh.loader import AggreateDataset, MeshDataset, normalise

# parser = argparse.ArgumentParser(
#     prog="Warpmesh", description="warp the mesh", epilog="warp the mesh"
# )
# parser.add_argument("-config", default="", type=str, required=True)
# args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# config_name = args.config

config_name = "MRT_miniset"
config = load_yaml_to_namespace(f"./configs/{config_name}")

# Define path where data get stored
now = datetime.now()
now_date = now.strftime("%Y-%m-%d-%H:%M_")
config.experiment_name = now_date + config_name

# Old dataset
data_root_old = "./data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=50_aniso_full_algo_6"
data_paths_old = [data_root_old]
print(f"Dataset old {data_paths_old}")

# New dataset
data_root_new = "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=50_aniso_full_meshtype_6"
data_paths_new = [data_root_new]
print(f"Dataset old {data_root_new}")

config.is_normalise = False


def create_dataset(config, data_paths):
    # Load datasets
    train_sets = [
        MeshDataset(
            os.path.join(data_path, "train"),
            transform=normalise if config.is_normalise else None,
            x_feature=config.x_feat,
            mesh_feature=config.mesh_feat,
            conv_feature=config.conv_feat,
            conv_feature_fix=config.conv_feat_fix,
            load_jacobian=config.use_jacob,
            use_cluster=config.use_cluster,
            r=config.cluster_r,
            load_analytical=True,
        )
        for data_path in data_paths
    ]

    test_sets = [
        MeshDataset(
            os.path.join(data_path, "test"),
            transform=normalise if config.is_normalise else None,
            x_feature=config.x_feat,
            mesh_feature=config.mesh_feat,
            conv_feature=config.conv_feat,
            conv_feature_fix=config.conv_feat_fix,
            load_jacobian=config.use_jacob,
            use_cluster=config.use_cluster,
            r=config.cluster_r,
            load_analytical=True,
        )
        for data_path in data_paths
    ]

    # val_sets = [
    #     MeshDataset(
    #         os.path.join(data_path, "val"),
    #         transform=normalise if config.is_normalise else None,
    #         x_feature=config.x_feat,
    #         mesh_feature=config.mesh_feat,
    #         conv_feature=config.conv_feat,
    #         conv_feature_fix=config.conv_feat_fix,
    #         load_jacobian=config.use_jacob,
    #         use_cluster=config.use_cluster,
    #         r=config.cluster_r,
    #         load_analytical=True,
    #     )
    #     for data_path in data_paths
    # ]

    # for training, datasets preperation
    train_set = AggreateDataset(train_sets)
    test_set = AggreateDataset(test_sets)
    # val_set = AggreateDataset(val_sets)

    # Loading and Batching
    train_loader = DataLoader(train_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)
    return train_set, test_set, train_loader, test_loader


train_set_old, test_set_old, train_loader_old, test_loader_old = create_dataset(
    config, data_paths_old
)
train_set_new, test_set_new, train_loader_new, test_loader_new = create_dataset(
    config, data_paths_new
)


num_selected = 10
print("Num selected: ", num_selected)
print("old ", train_set_old[num_selected])
print("new ", train_set_new[num_selected])
print("dist param old", train_set_old[num_selected].dist_params)
print("dist param new", train_set_new[num_selected].dist_params)

for i in range(6):
    print(
        f"Dim: {i} ",
        np.allclose(
            train_set_old[num_selected].mesh_feat[:, i],
            train_set_new[num_selected].mesh_feat[:, i],
        ),
    )


mesh_gen = wm.UnstructuredSquareMesh()

mesh_old = mesh_gen.load_mesh(
    file_path=os.path.join(f"{data_root_old}/mesh", f"mesh{num_selected}.msh")
)
mesh_old_fine = mesh_gen.load_mesh(
    file_path=os.path.join(f"{data_root_old}/mesh_fine", f"mesh{num_selected}.msh")
)

mesh_function_space_old = fd.Function(fd.FunctionSpace(mesh_old, "CG", 1))

mesh_new = mesh_gen.load_mesh(
    file_path=os.path.join(f"{data_root_new}/mesh", f"mesh_{num_selected:04d}.msh")
)
mesh_new_fine = mesh_gen.load_mesh(
    file_path=os.path.join(f"{data_root_new}/mesh_fine", f"mesh_{num_selected:04d}.msh")
)
mesh_function_space_new = fd.Function(fd.FunctionSpace(mesh_new, "CG", 1))

# ====  Plot mesh, solution, error ======================
rows, cols = 3, 4
fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), layout="compressed")
cmap = "seismic"

## Row-1
# High resolution mesh old
fd.triplot(mesh_old_fine, axes=ax[0, 0])
ax[0, 0].set_title("Fine mesh (old)")
# Orginal low resolution uniform mesh old
fd.triplot(mesh_old, axes=ax[0, 1])
ax[0, 1].set_title("Mesh (old)")
# Solution
mesh_function_space_old.dat.data[:] = (
    train_set_old[num_selected].mesh_feat[:, 2].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_old, cmap=cmap, axes=ax[0, 2])
ax[0, 2].set_title("Solution (Old)")
plt.colorbar(cb)
# Hessian norm
mesh_function_space_old.dat.data[:] = (
    train_set_old[num_selected].mesh_feat[:, 3].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_old, cmap=cmap, axes=ax[0, 3])
ax[0, 3].set_title("Hessian norm (Old)")
plt.colorbar(cb)


## Row-2
# High resolution mesh old
fd.triplot(mesh_new_fine, axes=ax[1, 0])
ax[1, 0].set_title("Fine mesh (old)")
# Orginal low resolution uniform mesh old
fd.triplot(mesh_new, axes=ax[1, 1])
ax[1, 1].set_title("Mesh (new)")
# Solution
mesh_function_space_new.dat.data[:] = (
    train_set_new[num_selected].mesh_feat[:, 2].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_new, cmap=cmap, axes=ax[1, 2])
ax[1, 2].set_title("Solution (new)")
plt.colorbar(cb)
# Hessian norm
mesh_function_space_new.dat.data[:] = (
    train_set_new[num_selected].mesh_feat[:, 3].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_new, cmap=cmap, axes=ax[1, 3])
ax[1, 3].set_title("Hessian norm (new)")
plt.colorbar(cb)


## Row-3
# High resolution mesh old
mesh_function_space_new.dat.data[:] = (
    train_set_old[num_selected].mesh_feat[:, 0].reshape(-1)[:]
    - train_set_new[num_selected].mesh_feat[:, 0].reshape(-1)[:]
)

cb = fd.tripcolor(mesh_function_space_new, cmap=cmap, axes=ax[2, 0])
ax[2, 0].set_title("Diff between Fine mesh (x-direction)")
plt.colorbar(cb)

# Orginal low resolution uniform mesh old
mesh_function_space_new.dat.data[:] = (
    train_set_old[num_selected].mesh_feat[:, 1].reshape(-1)[:]
    - train_set_new[num_selected].mesh_feat[:, 1].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_new, cmap=cmap, axes=ax[2, 1])
ax[2, 1].set_title("Diff between Fine mesh (y-direction)")
plt.colorbar(cb)

# Solution
mesh_function_space_new.dat.data[:] = (
    train_set_old[num_selected].mesh_feat[:, 2].reshape(-1)[:]
    - train_set_new[num_selected].mesh_feat[:, 2].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_new, cmap=cmap, axes=ax[2, 2])
ax[2, 2].set_title("Solution (u_old - u_new)")
plt.colorbar(cb)
# Hessian norm
mesh_function_space_new.dat.data[:] = (
    train_set_old[num_selected].mesh_feat[:, 3].reshape(-1)[:]
    - train_set_new[num_selected].mesh_feat[:, 3].reshape(-1)[:]
)
cb = fd.tripcolor(mesh_function_space_new, cmap=cmap, axes=ax[2, 3])
ax[2, 3].set_title("Hessian norm (h_old - h_new)")
plt.colorbar(cb)

# # Solution on high resolution mesh
# cb = fd.tripcolor(u_exact, cmap=cmap, axes=ax[1, 0])
# ax[1, 0].set_title(f"Solution on High Resolution (u_exact)")
# plt.colorbar(cb)
# # Solution on orginal low resolution uniform mesh
# cb = fd.tripcolor(u_og, cmap=cmap, axes=ax[1, 1])
# ax[1, 1].set_title(f"Solution on uniform Mesh")
# plt.colorbar(cb)
# # Solution on adapted mesh (MA)
# cb = fd.tripcolor(u_ma, cmap=cmap, axes=ax[1, 2])
# ax[1, 2].set_title(f"Solution on Adapted Mesh (MA)")
# plt.colorbar(cb)

# if u_model:
#     # Solution on adapted mesh (Model)
#     cb = fd.tripcolor(u_model, cmap=cmap, axes=ax[1, 3])
#     ax[1, 3].set_title(f"Solution on Adapted Mesh ({model_name})")
#     plt.colorbar(cb)

# err_orignal_mesh = fd.assemble(u_og - u_exact)
# err_adapted_mesh_ma = fd.assemble(u_ma - u_exact)

# if u_model:
#     err_adapted_mesh_model = fd.assemble(u_model - u_exact)
#     err_abs_max_val_adapted_mesh_model = max(
#         abs(err_adapted_mesh_model.dat.data[:].max()),
#         abs(err_adapted_mesh_model.dat.data[:].min()),
#     )
# else:
#     err_abs_max_val_adapted_mesh_model = 0.0

# err_abs_max_val_ori = max(
#     abs(err_orignal_mesh.dat.data[:].max()),
#     abs(err_orignal_mesh.dat.data[:].min()),
# )
# err_abs_max_val_adapted_ma = max(
#     abs(err_adapted_mesh_ma.dat.data[:].max()),
#     abs(err_adapted_mesh_ma.dat.data[:].min()),
# )

# err_abs_max_val = max(
#     max(err_abs_max_val_ori, err_abs_max_val_adapted_ma),
#     err_abs_max_val_adapted_mesh_model,
# )
# err_v_max = err_abs_max_val
# err_v_min = -err_v_max

# # Visualize the monitor values of MA
# monitor_val = raw_data.get("monitor_val")
# monitor_val_vis_holder = fd.Function(self.scalar_space)
# monitor_val_vis_holder.dat.data[:] = monitor_val[:, 0]

# # Error on high resolution mesh
# cb = fd.tripcolor(monitor_val_vis_holder, cmap=cmap, axes=ax[2, 0])
# ax[2, 0].set_title(f"Monitor Values")
# plt.colorbar(cb)
# # Monitor values for mesh movement
# cb = fd.tripcolor(
#     err_orignal_mesh,
#     cmap=cmap,
#     axes=ax[2, 1],
#     vmax=err_v_max,
#     vmin=err_v_min,
# )
# ax[2, 1].set_title(f"Error (u-u_exact) uniform Mesh | L2 Norm: {error_og:.5f}")
# plt.colorbar(cb)
# # Error on adapted mesh (MA)
# cb = fd.tripcolor(
#     err_adapted_mesh_ma,
#     cmap=cmap,
#     axes=ax[2, 2],
#     vmax=err_v_max,
#     vmin=err_v_min,
# )
# ax[2, 2].set_title(
#     f"Error (u-u_exact) MA| L2 Norm: {error_adapt:.5f} | {(error_og-error_adapt)/error_og*100:.2f}%"
# )
# plt.colorbar(cb)

# if u_model:
#     # Error on adapted mesh (Model)
#     cb = fd.tripcolor(
#         err_adapted_mesh_model,
#         cmap=cmap,
#         axes=ax[2, 3],
#         vmax=err_v_max,
#         vmin=err_v_min,
#     )
#     ax[2, 3].set_title(
#         f"Error (u-u_exact) {model_name}| L2 Norm: {error_model:.5f} | {(error_og-error_model)/error_og*100:.2f}%"
#     )
#     plt.colorbar(cb)

for rr in range(rows):
    for cc in range(cols):
        ax[rr, cc].set_aspect("equal", "box")


fig.savefig(f"view_data_diff_{num_selected:04d}.png")
