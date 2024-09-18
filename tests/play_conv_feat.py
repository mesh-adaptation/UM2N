import os

import firedrake as fd

import warpmesh as wm

os.environ["OMP_NUM_THREADS"] = "1"

num_grid_x = 20
num_grid_y = 20
scale_x = 1
scale_y = 1
max_dist = 0
n_dist = 16
z_max = 1
z_min = 0

mesh = fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y)
helmholtz_eq = wm.RandomHelmholtzGenerator(
    dist_params={
        "max_dist": max_dist,
        "n_dist": n_dist,
        "x_start": 0,
        "x_end": 1,
        "y_start": 0,
        "y_end": 1,
        "z_max": z_max,
        "z_min": z_min,
    }
)

res = helmholtz_eq.discretise(mesh)  # discretise the equation

solver = wm.HelmholtzSolver(
    params={
        "function_space": res["function_space"],
        "LHS": res["LHS"],
        "RHS": res["RHS"],
        "bc": res["bc"],
    }
)
uh = solver.solve_eq()
hessian = wm.MeshGenerator(
    params={
        "num_grid_x": num_grid_x,
        "num_grid_y": num_grid_y,
        "helmholtz_eq": helmholtz_eq,
        "mesh": fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y),
    }
).get_hessian(mesh)

hessian_norm = wm.MeshGenerator(
    params={
        "num_grid_x": num_grid_x,
        "num_grid_y": num_grid_y,
        "helmholtz_eq": helmholtz_eq,
        "mesh": fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y),
    }
).monitor_func(mesh)
hessian_norm = fd.project(hessian_norm, fd.FunctionSpace(mesh, "CG", 1))

func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
grad_uh_interpolate = fd.interpolate(fd.grad(uh), func_vec_space)

mesh_gen = wm.MeshGenerator(
    params={
        "num_grid_x": num_grid_x,
        "num_grid_y": num_grid_y,
        "helmholtz_eq": helmholtz_eq,
        "mesh": fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y),
    }
)

new_mesh = mesh_gen.move_mesh()

# solve the equation on the new mesh
new_res = helmholtz_eq.discretise(new_mesh)
new_solver = wm.HelmholtzSolver(
    params={
        "function_space": new_res["function_space"],
        "LHS": new_res["LHS"],
        "RHS": new_res["RHS"],
        "bc": new_res["bc"],
    }
)
uh_new = new_solver.solve_eq()

# process the data for training
mesh_processor = wm.MeshProcessor(
    original_mesh=mesh,
    optimal_mesh=new_mesh,
    function_space=new_res["function_space"],
    feature={
        "uh": uh.dat.data_ro.reshape(-1, 1),
        "grad_uh": grad_uh_interpolate.dat.data_ro.reshape(-1, 2),
        "hessian": hessian.dat.data_ro.reshape(-1, 4),
    },
    raw_feature={
        "uh": uh,
        "hessian_norm": hessian_norm,
    },
)

mesh_processor.get_conv_feat()
