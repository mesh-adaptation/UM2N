import time
from types import SimpleNamespace

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import UM2N
from inference_utils import InputPack, find_bd, find_edges, get_conv_feat

print("Setting up solver.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#################### Load trained model ####################

with open("./pretrain_model/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
    # print(config_data)

config = SimpleNamespace(**config_data)

# Append the monitor val at the end
# config.mesh_feat.append("monitor_val")
# config.mesh_feat = ["coord", "u", "monitor_val"]
config.mesh_feat = ["coord", "monitor_val"]

# print("# Evaluation Pipeline Started\n")
print(config)

model = UM2N.M2N_T(
    deform_in_c=config.num_deform_in,
    gfe_in_c=config.num_gfe_in,
    lfe_in_c=config.num_lfe_in,
)
model_file_path = "./pretrain_model/model_999.pth"
model = UM2N.load_model(model, model_file_path)
# model = load_model(run, config, epoch, "output_sim")
model.eval()
model = model.to(device)
###########################################################


# Simple Helmholtz equation
# =========================
#
# Let's start by considering the modified Helmholtz equation on a unit square,
# :math:`\Omega`, with boundary :math:`\Gamma`:
#
# .. math::
#
#    -\nabla^2 u + u &= f
#
#    \nabla u \cdot \vec{n} &= 0 \quad \textrm{on}\ \Gamma
#
# for some known function :math:`f`. The solution to this equation will
# be some function :math:`u\in V`, for some suitable function space
# :math:`V`, that satisfies these equations. Note that this is the
# Helmholtz equation that appears in meteorology, rather than the
# indefinite Helmholtz equation :math:`\nabla^2 u + u = f` that arises
# in wave problems.
#
# We transform the equation into weak form by multiplying by an arbitrary
# test function in :math:`V`, integrating over the domain and then
# integrating by parts. The variational problem so derived reads: find
# :math:`u \in V` such that:
#
# .. math::
#
#    \require{cancel}
#    \int_\Omega \nabla u\cdot\nabla v  + uv\ \mathrm{d}x = \int_\Omega
#    vf\ \mathrm{d}x + \cancel{\int_\Gamma v \nabla u \cdot \vec{n} \mathrm{d}s}
#
# Note that the boundary condition has been enforced weakly by removing
# the surface term resulting from the integration by parts.
#
# We can choose the function :math:`f`, so we take:
#
# .. math::
#
#    f = (1.0 + 8.0\pi^2)\cos(2\pi x)\cos(2\pi y)
#
# which conveniently yields the analytic solution:
#
# .. math::
#
#    u = \cos(2\pi x)\cos(2\pi y)
#
# However we wish to employ this as an example for the finite element
# method, so lets go ahead and produce a numerical solution.
#
# First, we always need a mesh. Let's have a :math:`10\times10` element unit square::


# These meshes can be replaced by generated mesh
mesh = fd.UnitSquareMesh(50, 50)
mesh_adapted = fd.UnitSquareMesh(50, 50)


def solve_helmholtz(mesh):
    # We need to decide on the function space in which we'd like to solve the
    # problem. Let's use piecewise linear functions continuous between
    # elements::

    V = fd.FunctionSpace(mesh, "CG", 1)

    # We'll also need the test and trial functions corresponding to this
    # function space::

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    # We declare a function over our function space and give it the
    # value of our right hand side function::

    f = fd.Function(V)
    x, y = fd.SpatialCoordinate(mesh)
    f.interpolate(
        (1 + 8 * fd.pi * fd.pi) * fd.cos(x * fd.pi * 2) * fd.cos(y * fd.pi * 2)
    )

    # We can now define the bilinear and linear forms for the left and right
    # hand sides of our equation respectively::

    a = (fd.inner(fd.grad(u), fd.grad(v)) + fd.inner(u, v)) * fd.dx
    L = fd.inner(f, v) * fd.dx

    # Finally we solve the equation. We redefine `u` to be a function
    # holding the solution::

    u = fd.Function(V)

    # Since we know that the Helmholtz equation is
    # symmetric, we instruct PETSc to employ the conjugate gradient method
    # and do not worry about preconditioning for the purposes of this demo ::

    fd.solve(a == L, u, solver_parameters={"ksp_type": "cg", "pc_type": "none"})

    f.interpolate(fd.cos(x * fd.pi * 2) * fd.cos(y * fd.pi * 2))
    print("L2 error ", fd.sqrt(fd.assemble(fd.dot(u - f, u - f) * fd.dx)))

    return mesh, V, u


mesh, V, u = solve_helmholtz(mesh)


def monitor_func(mesh, u, alpha=5.0):
    vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    uh_grad = fd.interpolate(fd.grad(u), vec_space)
    grad_norm = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    grad_norm.interpolate(uh_grad[0] ** 2 + uh_grad[1] ** 2)
    # normalizer = (grad_norm.vector().max() + 1e-6)
    # grad_norm.interpolate(alpha * grad_norm / normalizer + 1.0)
    return grad_norm


# Extract input features
coords = mesh.coordinates.dat.data_ro
print(f"coords {coords.shape}")
# print(f"conv feat {conv_feat.shape}")
edge_idx = find_edges(mesh, V)
print(f"edge idx {edge_idx.shape}")
bd_mask, _, _, _, _ = find_bd(mesh, V)
print(f"boundary mask {bd_mask.shape}")


monitor_val = monitor_func(mesh, u)
filter_monitor_val = np.minimum(1e3, monitor_val.dat.data[:])
filter_monitor_val = np.maximum(0, filter_monitor_val)
monitor_val.dat.data[:] = filter_monitor_val / filter_monitor_val.max()
conv_feat = get_conv_feat(mesh, monitor_val)
start_time = time.perf_counter()
sample = InputPack(
    coord=coords,
    monitor_val=monitor_val.dat.data_ro.reshape(-1, 1),
    edge_index=edge_idx,
    bd_mask=bd_mask,
    conv_feat=conv_feat,
    stack_boundary=False,
)
adapted_coord = model(sample)
end_time = time.perf_counter()
print(f"Model inference time: {(end_time - start_time)*1e3} ms")
# Update the mesh to adpated mesh
mesh_adapted.coordinates.dat.data[:] = adapted_coord.cpu().detach().numpy()

mesh_adapted, _, u_adapted = solve_helmholtz(mesh_adapted)


rows = 3
cols = 2
cmap = "seismic"
fig, ax = plt.subplots(rows, cols, figsize=(8, 12))

# Uniform mesh
fd.triplot(mesh, axes=ax[0, 0])
ax[0, 0].set_title("Original Mesh")
fd.tripcolor(u, axes=ax[0, 1], cmap=cmap)
ax[0, 1].set_title("u")

# Adapted mesh
fd.triplot(mesh_adapted, axes=ax[1, 0])
ax[1, 0].set_title("Adapated Mesh (UM2N)")
fd.tripcolor(u_adapted, axes=ax[1, 1], cmap=cmap)
ax[1, 1].set_title("u_adapted")

# Monitor val
fd.tripcolor(monitor_val, axes=ax[2, 0], cmap=cmap)
ax[2, 0].set_title("Monitor val")

plt.savefig("helm_example.png")
plt.show()
