import os
import wandb
import glob
import time
import torch
import yaml
import pickle
import firedrake as fd
import numpy as np
import warpmesh as wm
import matplotlib.pyplot as plt
from types import SimpleNamespace
from inference_utils import get_conv_feat, find_edges, find_bd, InputPack, load_model

print("Setting up solver.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#################### Load trained model ####################

with open(f"./pretrain_model/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
    # print(config_data)

config = SimpleNamespace(**config_data)

# Append the monitor val at the end
# config.mesh_feat.append("monitor_val")
# config.mesh_feat = ["coord", "u", "monitor_val"]
config.mesh_feat = ["coord", "monitor_val"]

# print("# Evaluation Pipeline Started\n")
print(config)

model = wm.M2N_T(
    deform_in_c=config.num_deform_in,
    gfe_in_c=config.num_gfe_in,
    lfe_in_c=config.num_lfe_in,
)
model_file_path = "./pretrain_model/model_999.pth"
model = wm.load_model(model, model_file_path)
# model = load_model(run, config, epoch, "output_sim")
model.eval()
model = model.to(device)
###########################################################


# physical constants
nu_val = 0.001
nu = fd.Constant(nu_val)

# time step
dt = 0.001
# define a firedrake constant equal to dt so that variation forms
# not regenerated if we change the time step
k = fd.Constant(dt)


RUN_SIM = True
VIZ = True
SAVE_DATA = True
# all_mesh_names = ["cylinder_010.msh", "cylinder_015.msh", "cylinder_020.msh", "cylinder_040.msh"]
# all_mesh_names = ["cylinder_020.msh"]
# all_mesh_names = ["cylinder_010.msh", "cylinder_015.msh"]
# all_mesh_names = ["cylinder_040.msh"]
# all_mesh_names = ["cylinder_030.msh", "cylinder_035.msh"]
# all_mesh_names = ["cylinder_020.msh", "cylinder_010.msh"]
all_mesh_names = ["cylinder_015.msh", "cylinder_010.msh"]
# all_mesh_names = ["cylinder_square.msh"]
# instead of using RectangleMesh, we now read the mesh from file
# mesh_name = "cylinder_010.msh"
# mesh_name = "cylinder_fine.msh"
# mesh_name = "cylinder_very_fine.msh"
# mesh_name = "cylinder_coarse.msh"

# mesh_name = "cylinder_multiple_very_fine.msh"
# mesh_name = "cylinder_multiple_fine.msh"
# mesh_name = "cylinder_multiple_coarse.msh"
for mesh_name in all_mesh_names:
    mesh_path = f"./meshes/{mesh_name}"
    mesh = fd.Mesh(mesh_path)
    adapted_mesh = fd.Mesh(mesh.coordinates.copy(deepcopy=True))
    init_coord = mesh.coordinates.copy(deepcopy=True).dat.data[:]
    u_grid = fd.Function(mesh.coordinates.function_space(), name="gridvelocity")

    V = fd.VectorFunctionSpace(mesh, "CG", 2)
    V_adapted = fd.VectorFunctionSpace(adapted_mesh, "CG", 2)

    Q = fd.FunctionSpace(mesh, "CG", 1)
    Q_adapted = fd.FunctionSpace(adapted_mesh, "CG", 1)

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    p = fd.TrialFunction(Q)
    q = fd.TestFunction(Q)

    vortex = fd.Function(Q)

    u_now = fd.Function(V)
    u_next = fd.Function(V)
    u_star = fd.Function(V)
    p_now = fd.Function(Q)
    p_next = fd.Function(Q)

    u_adapted = fd.Function(V_adapted)
    p_adapted = fd.Function(Q_adapted)

    # Expressions for the variational forms
    n = fd.FacetNormal(mesh)
    f = fd.Constant((0.0, 0.0))
    u_mid = 0.5 * (u_now + u)
    u_adv = u_now - u_grid

    def sigma(u, p):
        return 2 * nu * fd.sym(fd.nabla_grad(u)) - p * fd.Identity(len(u))

    x, y = fd.SpatialCoordinate(mesh)

    if "multiple" in mesh_name:
        # Define boundary conditions
        bcu = [
            fd.DirichletBC(
                V, fd.Constant((0, 0)), (1, 4, 5, 6, 7, 8)
            ),  # top-bottom and cylinder
            fd.DirichletBC(V, ((4.0 * 1.5 * y * (0.41 - y) / 0.41**2), 0), 2),
        ]  # inflow
    else:
        # Define boundary conditions
        bcu = [
            fd.DirichletBC(V, fd.Constant((0, 0)), (1, 4)),  # top-bottom and cylinder
            fd.DirichletBC(V, ((4.0 * 1.5 * y * (0.41 - y) / 0.41**2), 0), 2),
        ]  # inflow
    bcp = [fd.DirichletBC(Q, fd.Constant(0), 3)]  # outflow

    U_mean = 1.0
    re_num = int(U_mean * 0.1 / nu_val)
    print(f"Re = {re_num}")

    # Define variational forms
    F1 = (
        fd.inner((u - u_now) / k, v) * fd.dx
        + fd.inner(fd.dot(u_adv, fd.nabla_grad(u_mid)), v) * fd.dx
        + fd.inner(sigma(u_mid, p_now), fd.sym(fd.nabla_grad(v))) * fd.dx
        + fd.inner(p_now * n, v) * fd.ds
        - fd.inner(nu * fd.dot(fd.nabla_grad(u_mid), n), v) * fd.ds
        - fd.inner(f, v) * fd.dx
    )

    a1, L1 = fd.system(F1)

    a2 = fd.inner(fd.nabla_grad(p), fd.nabla_grad(q)) * fd.dx
    L2 = (
        fd.inner(fd.nabla_grad(p_now), fd.nabla_grad(q)) * fd.dx
        - (1 / k) * fd.inner(fd.div(u_star), q) * fd.dx
    )

    a3 = fd.inner(u, v) * fd.dx
    L3 = (
        fd.inner(u_star, v) * fd.dx
        - k * fd.inner(fd.nabla_grad(p_next - p_now), v) * fd.dx
    )

    # Define linear problems
    prob1 = fd.LinearVariationalProblem(a1, L1, u_star, bcs=bcu)
    prob2 = fd.LinearVariationalProblem(a2, L2, p_next, bcs=bcp)
    prob3 = fd.LinearVariationalProblem(a3, L3, u_next)

    # Define solvers
    solve1 = fd.LinearVariationalSolver(
        prob1, solver_parameters={"ksp_type": "gmres", "pc_type": "sor"}
    )
    solve2 = fd.LinearVariationalSolver(
        prob2, solver_parameters={"ksp_type": "cg", "pc_type": "gamg"}
    )
    solve3 = fd.LinearVariationalSolver(
        prob3, solver_parameters={"ksp_type": "cg", "pc_type": "sor"}
    )

    # Prep for saving solutions
    u_save = fd.Function(V).assign(u_now)
    p_save = fd.Function(Q).assign(p_now)
    exp_name = mesh_name.split(".msh")[0]
    outfile_u = fd.File(f"outputs_sim_ale/{exp_name}/u.pvd")
    outfile_u_grid = fd.File(f"outputs_sim_ale/{exp_name}/u_grid.pvd")
    outfile_p = fd.File(f"outputs_sim_ale/{exp_name}/p.pvd")
    outfile_u.write(u_save)
    outfile_u_grid.write(u_grid)
    outfile_p.write(p_save)

    # Time loop
    t = 0.0
    t_end = 8.0

    total_step = int((t_end - t) / dt)
    print("Beginning time loop...")

    def monitor_func(mesh, u, alpha=5.0):
        tensor_space = fd.TensorFunctionSpace(mesh, "CG", 1)
        uh_grad = fd.interpolate(fd.grad(u), tensor_space)
        grad_norm = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
        grad_norm.interpolate(
            uh_grad[0, 0] ** 2
            + uh_grad[0, 1] ** 2
            + uh_grad[1, 0] ** 2
            + uh_grad[1, 1] ** 2
        )
        # normalizer = (grad_norm.vector().max() + 1e-6)
        # grad_norm.interpolate(alpha * grad_norm / normalizer + 1.0)
        return grad_norm

    # Extract input features
    coords = mesh.coordinates.dat.data_ro
    print(f"coords {coords.shape}")
    # print(f"conv feat {conv_feat.shape}")
    edge_idx = find_edges(mesh, Q)
    print(f"edge idx {edge_idx.shape}")
    bd_mask, _, _, _, _ = find_bd(mesh, Q)
    print(f"boundary mask {bd_mask.shape}")

    u_list = []
    step_cnt = 0
    save_interval = 10
    total_step = 8000
    adapted_coord = torch.tensor(init_coord)
    monitor_val = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    output_path = f"outputs_sim_ale/{exp_name}/adapt_T_final/Re_{re_num}_total_{total_step}_save_{save_interval}"
    output_data_path = f"{output_path}/data"
    output_plot_path = f"{output_path}/plot"
    output_stat_path = f"{output_path}/stat"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_plot_path, exist_ok=True)
    os.makedirs(output_stat_path, exist_ok=True)

    if RUN_SIM:
        with torch.no_grad():
            while t < t_end:
                start_time = time.perf_counter()
                solve1.solve()
                solve2.solve()
                solve3.solve()
                end_time = time.perf_counter()
                print(f"PDE solve time: {(end_time - start_time)*1e3} ms")

                t += dt

                u_save.assign(u_next)
                p_save.assign(p_next)
                outfile_u.write(u_save)
                outfile_p.write(p_save)

                # u_list.append(fd.Function(u_next))

                # update solutions
                u_now.assign(u_next)
                p_now.assign(p_next)

                start_time = time.perf_counter()
                # Store the solutions to adapted meshes
                # so that we can safely modify mesh coordinates later
                end_time = time.perf_counter()
                print(f"Project before time: {(end_time - start_time)*1e3} ms")

                # TODO: interpolate might be faster however requries to update firedrake version
                # u_adapted.interpolate(u_next)
                # p_adapted.interpolate(p_next)

                if np.abs(t - np.round(t, decimals=0)) < 1.0e-8:
                    print("time = {0:.3f}".format(t))

                if step_cnt % save_interval == 0:
                    # print(f"{step_cnt} steps done.")
                    vorticity = vortex.project(fd.curl(u_now)).dat.data[:]
                    plot_dict = {}
                    plot_dict["mesh_original"] = init_coord
                    plot_dict["mesh_adapt"] = adapted_coord.cpu().detach().numpy()
                    plot_dict["u"] = u_now.dat.data[:]
                    plot_dict["p"] = p_now.dat.data[:]
                    plot_dict["monitor_val"] = monitor_val.dat.data[:]
                    plot_dict["vortex"] = vorticity
                    plot_dict["step"] = step_cnt
                    plot_dict["dt"] = dt
                    ret_file = f"{output_data_path}/data_{step_cnt:06d}.pkl"
                    if SAVE_DATA:
                        with open(ret_file, "wb") as file:
                            pickle.dump(plot_dict, file)
                    print(
                        f"{step_cnt} steps done. Max vorticity: {np.max(vorticity)}, Min vorticity: {np.min(vorticity)}"
                    )

                step_cnt += 1
                # Recover the mesh back to init coord
                mesh.coordinates.dat.data[:] = init_coord

                start_time = time.perf_counter()
                # Project u_adapted back to uniform mesh for computing monitors
                u_proj_from_adapted = fd.Function(V)
                u_adapted.dat.data[:] = u_next.dat.data[:]
                u_proj_from_adapted.project(u_adapted)
                end_time = time.perf_counter()
                print(f"Project for monitor time: {(end_time - start_time)*1e3} ms")

                start_time = time.perf_counter()
                monitor_val = monitor_func(mesh, u_proj_from_adapted)
                filter_monitor_val = np.minimum(1e3, monitor_val.dat.data[:])
                filter_monitor_val = np.maximum(0, filter_monitor_val)
                monitor_val.dat.data[:] = filter_monitor_val / filter_monitor_val.max()
                conv_feat = get_conv_feat(mesh, monitor_val)
                end_time = time.perf_counter()
                print(f"Compute monitor time: {(end_time - start_time)*1e3} ms")
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
                mesh.coordinates.dat.data[:] = adapted_coord.cpu().detach().numpy()
                u_grid.dat.data[:] = (
                    mesh.coordinates.dat.data[:] - adapted_mesh.coordinates.dat.data[:]
                ) / dt
                outfile_u_grid.write(u_grid)
                # Project the u_adapted and p_adapted to new adapted mesh for next timestep solving
                start_time = time.perf_counter()
                u_grid_max = np.abs(u_grid.dat.data).max()
                print(f"{u_grid_max=}")
                if u_grid_max < 5:
                    u_now.assign(u_next)
                    p_now.assign(p_next)
                else:
                    u_adapted.dat.data[:] = u_next.dat.data[:]
                    p_adapted.dat.data[:] = p_next.dat.data[:]
                    u_now.project(u_adapted)
                    p_now.project(p_adapted)
                    u_grid.dat.data[:] = 0.0
                end_time = time.perf_counter()
                print(f"Project after time: {(end_time - start_time)*1e3} ms")

                # TODO: interpolate might be faster however requries to update firedrake version
                # u_now.interpolate(u_adapted)
                # p_now.interpolate(p_adapted)

                # The buffer for adapted mesh should also be updated
                adapted_mesh.coordinates.dat.data[:] = (
                    adapted_coord.cpu().detach().numpy()
                )

                if step_cnt % total_step == 0:
                    break

        print("Simulation complete")

    all_data_files = sorted(glob.glob(f"{output_data_path}/*.pkl"))
    C_D_list = []
    C_L_list = []
    for idx, data_f in enumerate(all_data_files):
        with open(data_f, "rb") as f:
            data_dict = pickle.load(f)

            function_space = fd.FunctionSpace(mesh, "CG", 1)
            function_space_vec = fd.VectorFunctionSpace(mesh, "CG", 2)

            mesh_original = data_dict["mesh_original"]
            mesh_adapt = data_dict["mesh_adapt"]
            u = data_dict["u"]
            p = data_dict["p"]
            vorticity = data_dict["vortex"]
            monitor_val = data_dict["monitor_val"]

            # Mesh coordinates
            mesh.coordinates.dat.data[:] = init_coord
            adapted_mesh.coordinates.dat.data[:] = mesh_adapt

            # The velocity is extracted from adapted mesh, we need to define the function space based on adapted mesh
            function_space_adapted_vec = fd.VectorFunctionSpace(adapted_mesh, "CG", 2)
            function_space_adapted = fd.FunctionSpace(adapted_mesh, "CG", 1)
            u_holder = fd.Function(function_space_adapted_vec)
            u_holder.dat.data[:] = u

            p_holder = fd.Function(function_space_adapted)
            p_holder.dat.data[:] = p

            # The vorticity is extracted from adapted mesh, we need to define the function space based on adapted mesh
            vortex_holder = fd.Function(function_space_adapted)
            vortex_holder.dat.data[:] = vorticity

            Umean = 1.0
            L = 0.1
            n = fd.FacetNormal(adapted_mesh)
            F_D = fd.assemble(fd.dot(n, sigma(u_holder, p_holder))[0] * fd.ds(4))
            F_L = fd.assemble(fd.dot(n, sigma(u_holder, p_holder))[1] * fd.ds(4))
            C_D = 2 / (Umean**2 * L) * F_D
            C_L = 2 / (Umean**2 * L) * F_L
            C_D_list.append(C_D)
            C_L_list.append(C_L)

            if VIZ:
                rows = 4
                fig, ax = plt.subplots(rows, 1, figsize=(16, 20))
                # Uniform mesh
                fd.triplot(mesh, axes=ax[0])
                ax[0].set_title("Original Mesh")
                # Adapted mesh
                fd.triplot(adapted_mesh, axes=ax[1])
                ax[1].set_title("Adapated Mesh (UM2N)")

                cmap = "seismic"

                # p_holder = fd.Function(function_space)
                # p_holder.dat.data[:] = p
                # ax1 = ax[2]
                # ax1.set_xlabel('$x$', fontsize=16)
                # ax1.set_ylabel('$y$', fontsize=16)
                # ax1.set_title('FEM Navier-Stokes - channel flow - pressure', fontsize=16)
                # fd.tripcolor(p_holder ,axes=ax1, cmap=cmap)
                # # ax1.axis('equal')

                ax1 = ax[2]
                ax1.set_xlabel("$x$", fontsize=16)
                ax1.set_ylabel("$y$", fontsize=16)
                ax1.set_title("Navier-Stokes - channel flow - vorticity", fontsize=16)
                fd.tripcolor(vortex_holder, axes=ax1, cmap=cmap, vmax=100, vmin=-100)

                ax2 = ax[3]
                ax2.set_xlabel("$x$", fontsize=16)
                ax2.set_ylabel("$y$", fontsize=16)
                ax2.set_title("Navier-Stokes - channel flow - velocity", fontsize=16)
                cb = fd.tripcolor(u_holder, axes=ax2, cmap=cmap)
                # plt.colorbar(cb)
                # ax2.axis('equal')

                # monitor_holder = fd.Function(function_space)
                # monitor_holder.dat.data[:] = monitor_val
                # ax3 = ax[4]
                # ax3.set_xlabel('$x$', fontsize=16)
                # ax3.set_ylabel('$y$', fontsize=16)
                # ax3.set_title('Navier-Stokes - channel flow - Monitor Values', fontsize=16)
                # cb = fd.tripcolor(monitor_holder,axes=ax3, cmap=cmap)
                # plt.colorbar(cb)
                # ax3.axis('equal')

                for rr in range(rows):
                    ax[rr].set_aspect("equal", "box")
                # plt.tight_layout()
                plt.savefig(
                    f"{output_plot_path}/cylinder_Re_{re_num}_{idx:06d}_adapt.png"
                )
                plt.close()
            print(f"Idx {idx} Done")

    import pandas as pd

    df_stat = pd.DataFrame({"C_D": C_D_list, "C_L": C_L_list})
    df_stat.to_csv(f"{output_stat_path}/df_stat.csv")
