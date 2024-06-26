import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from warpmesh.loader import MeshDataset, normalise, AggreateDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_graph



def interpolate(u, ori_mesh_x, ori_mesh_y, moved_x, moved_y):
    """
    u: [bs, node_num, 1]
    ori_mesh_x: [bs, node_num, 1]
    ori_mesh_y: [bs, node_num, 1]
    moved_x: [bs, node_num, 1]
    moved_y: [bs, node_num, 1]

    Note: node_num equals to sample_num
    """
    batch_size = u.shape[0]
    sample_num = u.shape[1]
    # print(f"batch size: {batch_size}, sample num: {sample_num}")
    u_interpolateds = []
    for bs in range(batch_size):
        # For a sample point of interest, we need to do a weighted summation over all other sample points
        # To avoid using a loop, we expand an additonal dim of size sample_num
        original_mesh = torch.cat((ori_mesh_x[bs], ori_mesh_y[bs]), dim=-1)
        moved_mesh = torch.cat((moved_x[bs], moved_y[bs]), dim=-1).unsqueeze(-2).repeat(1, sample_num, 1)
        # print(f"new mesh shape {moved_mesh.shape}, original mesh shape {original_mesh.shape}")
        # print((moved_mesh - original_mesh),(moved_mesh - original_mesh).shape)
        # print("check dimension ", (moved_mesh - original_mesh)[:, 0])

        # The second dimension of distance is the different sample points
        distance = -torch.norm(moved_mesh - original_mesh, dim=-1) * np.sqrt(sample_num)
        # print('raw distance ', torch.norm(moved_mesh - original_mesh, dim=-1))
        # print('distance ', torch.norm(moved_mesh - original_mesh, dim=-1)* np.sqrt(sample_num))
        normalize = nn.Softmax(dim=-1)
        weight = normalize(distance)
        # print('weight shape ', weight.shape, u[bs].shape)
        # print('weight ', weight, u, u[bs].permute(1, 0) * weight)
        # print(u.shape, weight.shape)
        u_interpolateds.append(torch.sum(u[bs].permute(1, 0) * weight, dim=-1).unsqueeze(-1))
        # print(f"interpolated shape: {u_interpolateds[-1]}")
        # print('inte ', u_interpolated)
    return torch.stack(u_interpolateds, dim=0)


def compute_finite_difference(field):
    # Field: [bs, x_shape, y_shape]
    f_x = torch.zeros_like(field)
    f_x[:,:-1,:] = torch.diff(field, dim=-2)
    f_x[:,-1,:] = f_x[:,-2,:]

    f_y = torch.zeros_like(field)
    f_y[:,:,:-1] = torch.diff(field, dim=-1)
    f_y[:,:,-1] = f_y[:,:,-2]

    inv_dx = field.shape[-2] - 1
    inv_dy = field.shape[-1] - 1
    return f_x * inv_dx, f_y * inv_dy


def generate_samples_structured_grid(coords, field, grid_resolution=100, device='cpu'):
    num_meshes = coords.shape[0]
    nx = grid_resolution
    ny = grid_resolution
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    uniform_grid = torch.tensor(np.array(np.meshgrid(x, y)), dtype=torch.float)\
                    .reshape(1, 2, -1).repeat(num_meshes, 1, 1).permute(0, 2, 1).to(device)

    field_input = field.view(num_meshes, -1, field.shape[-1])
    coords_x = coords[: ,: ,0].unsqueeze(-1)
    coords_y = coords[: ,: ,1].unsqueeze(-1)
    new_meshes_x = uniform_grid[:, :, 0].unsqueeze(-1)
    new_meshes_y = uniform_grid[:, :, 1].unsqueeze(-1)

    # Interpolate to dense structured grid
    field = interpolate(field_input, coords_x, coords_y, new_meshes_x, new_meshes_y)
    field_x_, field_y_ = compute_finite_difference(field.view(field.shape[0], grid_resolution, grid_resolution))
    field_x_ = field_x_.view(num_meshes, -1, 1)
    field_y_ = field_y_.view(num_meshes, -1, 1)

    # Interpolate back to original mesh
    field_x = interpolate(field_x_, new_meshes_x, new_meshes_y, coords_x, coords_y)
    field_y = interpolate(field_y_, new_meshes_x, new_meshes_y, coords_x, coords_y)

    return uniform_grid, field, field_x_, field_y_, field_x, field_y

data_paths = ["./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6"]


conv_feat = ["conv_uh", "conv_hessian_norm"]
conv_feat_fix = ["conv_uh_fix"]

x_feat = ["coord", "bd_mask"]
mesh_feat = ["coord", "u", "hessian_norm", "grad_u"]


train_sets = [MeshDataset(
    os.path.join(data_path, "train"),
    transform=normalise,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
    conv_feature_fix=conv_feat_fix,
    load_jacobian=False,
    use_cluster=False,
    r=0.35,
) for data_path in data_paths]


batch_size = 2
train_set = AggreateDataset(train_sets)
train_loader = DataLoader(train_set, batch_size=batch_size)


cnt = 0
sample = None
for batch in train_loader:
    sample = batch
    break
print(sample)
coords = sample.mesh_feat.view(batch_size, -1, sample.mesh_feat.shape[-1])[:, :, :2]
solution = sample.mesh_feat.view(batch_size, -1, sample.mesh_feat.shape[-1])[:, :, 2].unsqueeze(-1)
hessian_norm = sample.mesh_feat.view(batch_size, -1, sample.mesh_feat.shape[-1])[:, :, 3].unsqueeze(-1)
print(f"coords: {coords.shape}, solution: {solution.shape}, hessian norm: {hessian_norm.shape}")

num_nodes = coords.shape[1]
num_samples = 5

grid_resolution = 100
meshes, solution_struct_grid, solution_x_strut_grid, solution_y_strut_grid, solution_x, solution_y  = generate_samples_structured_grid(coords, solution, grid_resolution)
solution_x = solution_x.view(batch_size, -1, 1)
solution_y = solution_y.view(batch_size, -1, 1)
print(f"Sampled meshes: {meshes.shape}, solution: {solution_struct_grid.shape}, solution_x: {solution_x.shape}, solution_y: {solution_y.shape}")


num_show = 1
num_variables = 4 # meshes, solution, solution_x, solution_y
fig, ax = plt.subplots(num_variables, num_show + 1, figsize=(4*(num_show + 1), 4 * num_variables))
ax[0, 0].scatter(coords[0,:,0], coords[0,:,1])
ax[0, 0].set_title(r"$\xi_{query}$")
ax[0, 1].scatter(meshes[0,:,0], meshes[0,:,1])
title_str = f"xi_f"
ax[0, 1].set_title(r"$\{}$".format(title_str))

ax[1, 0].scatter(coords[0,:,0], coords[0,:,1], c=solution[0,:,0])
ax[1, 0].set_title(r"$u_{query}$")
ax[1, 1].scatter(meshes[0,:,0], meshes[0,:,1], c=solution_struct_grid[0,:,0])
title_str_1 = f"u_f"
ax[1, 1].set_title(r"${}$".format(title_str_1))

ax[2, 0].scatter(coords[0,:,0], coords[0,:,1], c=solution_x[0,:,0])
ax[2, 0].set_title(r"$u_x$")
ax[2, 1].scatter(meshes[0,:,0], meshes[0,:,1], c=solution_x_strut_grid[0,:,0])
title_str_2 = f"u_x"
ax[2, 1].set_title(r"${}$".format(title_str_2))

ax[3, 0].scatter(coords[0,:,0], coords[0,:,1], c=solution_y[0,:,0])
ax[3, 0].set_title(r"$u_y$")
ax[3, 1].scatter(meshes[0,:,0], meshes[0,:,1], c=solution_y_strut_grid[0,:,0])
title_str_3 = f"u_y"
ax[3, 1].set_title(r"${}$".format(title_str_3))

# for i in range(1, num_show+1):
#     ax[0, i].scatter(meshes[i,:,0], meshes[i,:,1])
#     title_str = f"xi_f^{i}"
#     ax[0, i].set_title(r"$\{}$".format(title_str))

#     ax[1, i].scatter(meshes[i,:,0], meshes[i,:,1], c=solution_struct_grid[i,:,0])
#     title_str_1 = f"u_f^{i}"
#     ax[1, i].set_title(r"${}$".format(title_str_1))

#     ax[2, i].scatter(meshes[i,:,0], meshes[i,:,1], c=solution_x_strut_grid[i,:,0])
#     title_str_2 = f"u_x^{i}"
#     ax[2, i].set_title(r"${}$".format(title_str_2))

#     ax[3, i].scatter(meshes[i,:,0], meshes[i,:,1], c=solution_y_strut_grid[i,:,0])
#     title_str_3 = f"u_y^{i}"
#     ax[3, i].set_title(r"${}$".format(title_str_3))
plt.savefig("sampled_structure_grid.png")


