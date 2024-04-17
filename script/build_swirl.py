# Author: Chunyang Wang
# GitHub Username: chunyang-w

import os
import pandas as pd
import warpmesh as wm
import firedrake as fd
import shutil
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--mesh_type", type=int, default=6, help="algorithm used to generate mesh"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=(0.05 / 3),
        help="sigma used to control the initial ring shape",
    )
    parser.add_argument(
        "--r_0", type=float, default=0.2, help="radius of the initial ring"
    )
    parser.add_argument(
        "--x_0", type=float, default=0.5, help="center of the ring in x"
    )
    parser.add_argument(
        "--y_0", type=float, default=0.5, help="center of the ring in y"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="scalar coefficient of the swirl (velocity)",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5, help="interval for stroing sample file"
    )
    parser.add_argument(
        "--lc",
        type=float,
        default=5e-2,
        help="the length characteristic of the elements in the\
                            mesh (if using unstructured mesh)",
    )
    parser.add_argument(
        "--n_grid",
        type=int,
        default=20,
        help="number of grids in a mesh (only appliable when\
                                mesh_type is 0)",
    )
    parser.add_argument(
        "--n_monitor_smooth",
        type=int,
        default=10,
        help="number of times for applying a Laplacian smoother for monitor function",
    )
    args_ = parser.parse_args()
    print(args_)
    return args_


args = arg_parse()

mesh_type = args.mesh_type

# ====  Parameters ======================
problem = "swirl"

# simulation time & time steps
T = 1
n_step = 500 * 2
dt = T / n_step

# mesh setup
lc = args.lc
# n_grid = args.n_grid
n_grid = int(1 / lc)

# number of times for applying a Laplacian smoother for monitor function
n_monitor_smooth = args.n_monitor_smooth

# parameters for domain scale
scale_x = 1
scale_y = 1

# params for initial condition
sigma = args.sigma
r_0 = args.r_0
alpha = args.alpha
x_0 = args.x_0
y_0 = args.y_0

# params for stroing files
save_interval = args.save_interval
# list storing failing dts
fail_t = []

# =======================================


def move_data(target, source, start, num_file):
    if not os.path.exists(target):
        os.makedirs(target)
    else:
        # delete all files under the directory
        filelist = [f for f in os.listdir(target)]
        for f in filelist:
            os.remove(os.path.join(target, f))
    # copy data from data dir to train dir
    for i in range(start, num_file):
        shutil.copy(
            os.path.join(source, "data_{}.npy".format(i)),
            os.path.join(target, "data_{}.npy".format(i)),
        )


project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
dataset_dir = os.path.join(
    project_dir, "data", f"dataset_meshtype_{mesh_type}", problem
)  # noqa
problem_specific_dir = os.path.join(
    dataset_dir,
    f"sigma_{sigma:.3f}_alpha_{alpha}_r0_{r_0}_x0_{x_0}_y0_{y_0}_lc_{lc}_ngrid_{n_grid}_interval_{save_interval}_meshtype_{mesh_type}_smooth_{n_monitor_smooth}",
)  # noqa


problem_data_dir = os.path.join(problem_specific_dir, "data")
problem_plot_dir = os.path.join(problem_specific_dir, "plot")
problem_plot_compare_dir = os.path.join(problem_specific_dir, "plot_compare")
problem_log_dir = os.path.join(problem_specific_dir, "log")
problem_mesh_dir = os.path.join(problem_specific_dir, "mesh")
problem_mesh_fine_dir = os.path.join(problem_specific_dir, "mesh_fine")


if not os.path.exists(problem_data_dir):
    os.makedirs(problem_data_dir)
else:
    # delete all files under the directory
    filelist = [f for f in os.listdir(problem_data_dir)]
    for f in filelist:
        os.remove(os.path.join(problem_data_dir, f))

if not os.path.exists(problem_plot_dir):
    os.makedirs(problem_plot_dir)
else:
    # delete all files under the directory
    filelist = [f for f in os.listdir(problem_plot_dir)]
    for f in filelist:
        os.remove(os.path.join(problem_plot_dir, f))

if not os.path.exists(problem_plot_compare_dir):
    os.makedirs(problem_plot_compare_dir)
else:
    # delete all files under the directory
    filelist = [f for f in os.listdir(problem_plot_compare_dir)]
    for f in filelist:
        os.remove(os.path.join(problem_plot_compare_dir, f))

if not os.path.exists(problem_log_dir):
    os.makedirs(problem_log_dir)
else:
    # delete all files under the directory
    filelist = [f for f in os.listdir(problem_log_dir)]
    for f in filelist:
        os.remove(os.path.join(problem_log_dir, f))

if not os.path.exists(problem_mesh_dir):
    os.makedirs(problem_mesh_dir)
else:
    # delete all files under the directory
    filelist = [f for f in os.listdir(problem_mesh_dir)]
    for f in filelist:
        os.remove(os.path.join(problem_mesh_dir, f))

if not os.path.exists(problem_mesh_fine_dir):
    os.makedirs(problem_mesh_fine_dir)
else:
    # delete all files under the directory
    filelist = [f for f in os.listdir(problem_mesh_fine_dir)]
    for f in filelist:
        os.remove(os.path.join(problem_mesh_fine_dir, f))

i = 0


def fail_callback(t):
    """
    Call back for failing cases.
    Log current time for those cases which MA did not converge.
    """
    fail_t.append(t)


def sample_from_loop(
    uh,
    uh_grad,
    hessian,
    grad_u_norm,
    hessian_norm,
    monitor_values,
    phi,
    grad_phi,
    jacobian,
    jacobian_det,
    uh_new,
    mesh_og,
    mesh_new,
    function_space,
    function_space_fine,
    uh_fine,
    dur,
    sigma,
    alpha,
    r_0,
    t,
    error_og_list=[],
    error_adapt_list=[],
):
    """
    Call back function for storing data.
    """
    global i
    print("before processing")
    mesh_processor = wm.MeshProcessor(
        original_mesh=mesh_og,
        optimal_mesh=mesh_new,
        function_space=function_space,
        use_4_edge=True,
        feature={
            "uh": uh.dat.data_ro.reshape(-1, 1),
            "grad_uh": uh_grad.dat.data_ro.reshape(-1, 2),
            "grad_uh_norm": grad_u_norm.dat.data_ro.reshape(-1, 1),
            "hessian": hessian.dat.data_ro.reshape(-1, 4),
            "hessian_norm": hessian_norm.dat.data_ro.reshape(-1, 1),
            "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
            "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
            "phi": phi.dat.data_ro.reshape(-1, 1),
            "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
            "monitor_val": monitor_values.dat.data_ro.reshape(-1, 1),
        },
        raw_feature={
            "uh": uh,
            "grad_uh_norm": grad_u_norm,
            "hessian_norm": hessian_norm,
            "monitor_val": monitor_values,
            "jacobian": jacobian,
            "jacobian_det": jacobian_det,
        },
        swirl_params={
            "t": t,
            "sigma": sigma,
            "alpha": alpha,
            "r_0": r_0,
            "x_0": x_0,
            "y_0": y_0,
        },
        dur=dur,
    )

    mesh_processor.save_taining_data(os.path.join(problem_data_dir, f"data_{i:04d}"))

    # # ====  Plot Scripts ======================
    # fig = plt.figure(figsize=(15, 10))
    # ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # # Plot the exact solution
    # ax1.set_title('Solution field (HR)')
    # fd.trisurf(uh_fine, axes=ax1)
    # # Plot the solved solution
    # ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    # ax2.set_title('Solution field (Original Mesh)')
    # fd.trisurf(uh, axes=ax2)

    # ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    # ax3.set_title('Solution field (Adapted Mesh)')
    # fd.trisurf(uh_new, axes=ax3)

    # # Plot the mesh
    # ax4 = fig.add_subplot(2, 3, 4)
    # ax4.set_title('Original Mesh ')
    # fd.triplot(mesh_og, axes=ax4)

    # ax5 = fig.add_subplot(2, 3, 5)
    # ax5.set_title('Optimal Mesh')
    # # fd.tripcolor(
    # #     uh, cmap='coolwarm', axes=ax5)
    # fd.triplot(mesh_new, axes=ax5)

    # # plot mesh with function evaluated on it
    # ax6 = fig.add_subplot(2, 3, 6)
    # ax6.set_title('Solution Projected on Optimal Mesh')
    # fd.tripcolor(
    #     uh_new, cmap='coolwarm', axes=ax6)
    # fd.triplot(mesh_new, axes=ax6)

    # fig.savefig(
    #     os.path.join(
    #         problem_plot_dir, f"plot_{i:04d}.png")
    # )
    # plt.close()
    # fig, ax = plt.subplots()
    # ax.set_title("adapt error list")
    # ax.plot(error_adapt_list, linestyle='--', color='blue', label='adapt')
    # # ax.plot(error_og_list, linestyle='--', color='red', label='og')
    # ax.legend()
    # plt.show()

    # ==========================================
    # function_space_fine = fd.FunctionSpace(mesh_fine, 'CG', 1)
    uh_proj = fd.project(uh, function_space_fine)
    uh_new_proj = fd.project(uh_new, function_space_fine)

    error_original_mesh = fd.errornorm(uh_proj, uh_fine, norm_type="L2")
    error_optimal_mesh = fd.errornorm(uh_new_proj, uh_fine, norm_type="L2")
    df = pd.DataFrame(
        {
            "error_og": error_original_mesh,
            "error_adapt": error_optimal_mesh,
            "time": dur,
        },
        index=[0],
    )
    df.to_csv(os.path.join(problem_log_dir, f"log{i:04d}.csv"))
    print("error og/optimal:", error_original_mesh, error_optimal_mesh)

    # ====  Plot mesh, solution, error ======================
    rows, cols = 3, 3
    fig, ax = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), layout="compressed"
    )

    # High resolution mesh
    fd.triplot(mesh_fine, axes=ax[0, 0])
    ax[0, 0].set_title(f"High resolution Mesh (100 x 100)")
    # Orginal low resolution uniform mesh
    fd.triplot(mesh_og, axes=ax[0, 1])
    ax[0, 1].set_title(f"Original uniform Mesh")
    # Adapted mesh
    fd.triplot(mesh_new, axes=ax[0, 2])
    ax[0, 2].set_title(f"Adapted Mesh (MA)")

    cmap = "seismic"
    # Solution on high resolution mesh
    cb = fd.tripcolor(uh_fine, cmap=cmap, axes=ax[1, 0])
    ax[1, 0].set_title(f"Solution on High Resolution (u_exact)")
    plt.colorbar(cb)
    # Solution on orginal low resolution uniform mesh
    cb = fd.tripcolor(uh, cmap=cmap, axes=ax[1, 1])
    ax[1, 1].set_title(f"Solution on uniform Mesh")
    plt.colorbar(cb)
    # Solution on adapted mesh
    cb = fd.tripcolor(uh_new, cmap=cmap, axes=ax[1, 2])
    ax[1, 2].set_title(f"Solution on Adapted Mesh (MA)")
    plt.colorbar(cb)

    err_orignal_mesh = fd.assemble(uh_proj - uh_fine)
    err_adapted_mesh = fd.assemble(uh_new_proj - uh_fine)
    err_abs_max_val_ori = max(
        abs(err_orignal_mesh.dat.data[:].max()), abs(err_orignal_mesh.dat.data[:].min())
    )
    err_abs_max_val_adapted = max(
        abs(err_adapted_mesh.dat.data[:].max()), abs(err_adapted_mesh.dat.data[:].min())
    )
    err_abs_max_val = max(err_abs_max_val_ori, err_abs_max_val_adapted)
    err_v_max = err_abs_max_val
    err_v_min = -err_v_max

    # # Error on high resolution mesh
    # cb = fd.tripcolor(fd.assemble(uh_fine - uh_fine), cmap=cmap, axes=ax[2, 0], vmax=err_v_max, vmin=err_v_min)
    # ax[2, 0].set_title(f"Error Map High Resolution")
    # plt.colorbar(cb)

    # Monitor values
    cb = fd.tripcolor(monitor_values, cmap=cmap, axes=ax[2, 0])
    ax[2, 0].set_title(f"Monitor Values")
    plt.colorbar(cb)

    # Error on orginal low resolution uniform mesh
    cb = fd.tripcolor(
        err_orignal_mesh, cmap=cmap, axes=ax[2, 1], vmax=err_v_max, vmin=err_v_min
    )
    ax[2, 1].set_title(
        f"Error (u-u_exact) uniform Mesh | L2 Norm: {error_original_mesh:.5f}"
    )
    plt.colorbar(cb)
    # Error on adapted mesh
    cb = fd.tripcolor(
        err_adapted_mesh, cmap=cmap, axes=ax[2, 2], vmax=err_v_max, vmin=err_v_min
    )
    ax[2, 2].set_title(
        f"Error (u-u_exact) Adapted Mesh (MA)| L2 Norm: {error_optimal_mesh:.5f} | {(error_original_mesh-error_optimal_mesh)/error_original_mesh*100:.2f}%"
    )
    plt.colorbar(cb)

    for rr in range(rows):
        for cc in range(cols):
            ax[rr, cc].set_aspect("equal", "box")

    fig.savefig(os.path.join(problem_plot_compare_dir, f"plot_{i:04d}.png"))
    plt.close()
    i += 1
    return


# ====  Data Generation Scripts ======================
if __name__ == "__main__":
    print("In build_dataset.py")
    mesh = None
    mesh_fine = None
    mesh_new = None
    if mesh_type != 0:
        mesh_gen = wm.UnstructuredSquareMesh(mesh_type=mesh_type)
        mesh = mesh_gen.get_mesh(
            res=lc, file_path=os.path.join(problem_mesh_dir, "mesh.msh")
        )
        mesh_new = mesh_gen.get_mesh(
            res=lc, file_path=os.path.join(problem_mesh_dir, "mesh.msh")
        )
        mesh_gen_fine = wm.UnstructuredSquareMesh(mesh_type=mesh_type)
        mesh_fine = mesh_gen_fine.get_mesh(
            res=1e-2, file_path=os.path.join(problem_mesh_fine_dir, "mesh.msh")
        )
    else:
        mesh = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_new = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_fine = fd.UnitSquareMesh(100, 100)

    df = pd.DataFrame(
        {
            "sigma": [sigma],
            "alpha": [alpha],
            "r_0": [r_0],
            "x_0": [x_0],
            "y_0": [y_0],
            "save_interval": [save_interval],
            "T": [T],
            "n_step": [n_step],
            "dt": [dt],
            "fail_t": [fail_t],
            "lc": [lc],
            "num_fail_cases": [len(fail_t)],
            "mesh_type": [mesh_type],
        }
    )

    df.to_csv(os.path.join(problem_specific_dir, "info.csv"))

    # solver defination
    swril_solver = wm.SwirlSolver(
        mesh,
        mesh_fine,
        mesh_new,
        sigma=sigma,
        alpha=alpha,
        r_0=r_0,
        x_0=x_0,
        y_0=y_0,
        save_interval=save_interval,
        T=T,
        n_step=n_step,
        n_monitor_smooth=n_monitor_smooth,
    )

    swril_solver.solve_problem(callback=sample_from_loop, fail_callback=fail_callback)

    print("Done!")
# ====  Data Generation Scripts ======================
