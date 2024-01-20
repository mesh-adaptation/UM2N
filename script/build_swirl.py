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
    parser.add_argument('--mesh_type', type=int, default=6,
                        help='algorithm used to generate mesh')
    parser.add_argument('--sigma', type=float, default=(0.05/3),
                        help='sigma used to control the initial ring shape')
    parser.add_argument('--r_0', type=float, default=0.2,
                        help='radius of the initial ring')
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='scalar coefficient of the swirl (velocity)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='interval for stroing sample file')
    parser.add_argument('--lc', type=float, default=4.5e-2,
                        help='the length characteristic of the elements in the\
                            mesh (if using unstructured mesh)')
    args_ = parser.parse_args()
    print(args_)
    return args_


args = arg_parse()

mesh_type = args.mesh_type

# ====  Parameters ======================
problem = "swirl"

# simulation time & time steps
T = 1
n_step = 500
dt = T / n_step

# mesh setup
lc = args.lc

# parameters for domain scale
scale_x = 1
scale_y = 1

# params for initial condition
sigma = args.sigma
r_0 = args.r_0
alpha = args.alpha

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
            os.path.join(target, "data_{}.npy".format(i))
        )


project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
dataset_dir = os.path.join(project_dir, "data", f"dataset_meshtype_{mesh_type}", problem)  # noqa
problem_specific_dir = os.path.join(
        dataset_dir,
        f"sigma_{sigma:.3f}_alpha_{alpha}_r0_{r_0}_lc_{lc}_interval_{save_interval}_meshtype_{mesh_type}")  # noqa


problem_data_dir = os.path.join(problem_specific_dir, "data")
problem_plot_dir = os.path.join(problem_specific_dir, "plot")
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


def sample_from_loop(uh, uh_grad, hessian, hessian_norm,
                     phi, grad_phi,
                     jacobian, jacobian_det,
                     uh_new, mesh_og, mesh_new,
                     function_space,
                     function_space_fine,
                     uh_fine, dur,
                     sigma, alpha, r_0, t,
                     error_og_list=[],
                     error_adapt_list=[],
                     ):
    """
    Call back function for storing data.
    """
    global i
    print("before processing")
    mesh_processor = wm.MeshProcessor(
        original_mesh=mesh_og, optimal_mesh=mesh_new,
        function_space=function_space,
        use_4_edge=True,
        feature={
            "uh": uh.dat.data_ro.reshape(-1, 1),
            "grad_uh": uh_grad.dat.data_ro.reshape(
                -1, 2),
            "hessian": hessian.dat.data_ro.reshape(
                -1, 4),
            "hessian_norm": hessian_norm.dat.data_ro.reshape(
                -1, 1),
            "jacobian": jacobian.dat.data_ro.reshape(
                -1, 4),
            "jacobian_det": jacobian_det.dat.data_ro.reshape(
                -1, 1),
            "phi": phi.dat.data_ro.reshape(
                -1, 1),
            "grad_phi": grad_phi.dat.data_ro.reshape(
                -1, 2),
        },
        raw_feature={
            "uh": uh,
            "hessian_norm": hessian_norm,
            "jacobian": jacobian,
            "jacobian_det": jacobian_det,
        },
        swirl_params={
            "t": t,
            "sigma": sigma,
            "alpha": alpha,
            "r_0": r_0,
        },
        dur=dur
    )

    mesh_processor.save_taining_data(
        os.path.join(problem_data_dir, "data_{}".format(i))
    )

    # ====  Plot Scripts ======================
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # Plot the exact solution
    ax1.set_title('Solution field (HR)')
    fd.trisurf(uh_fine, axes=ax1)
    # Plot the solved solution
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.set_title('Solution field (Original Mesh)')
    fd.trisurf(uh, axes=ax2)

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.set_title('Solution field (Adapted Mesh)')
    fd.trisurf(uh_new, axes=ax3)

    # Plot the mesh
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_title('Hessian Norm')
    fd.trisurf(hessian_norm, axes=ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title('Original Mesh')
    fd.tripcolor(
        uh, cmap='coolwarm', axes=ax5)
    fd.triplot(mesh_og, axes=ax5)

    # plot mesh with function evaluated on it
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title('Adapted Mesh')
    fd.tripcolor(
        uh_new, cmap='coolwarm', axes=ax6)
    fd.triplot(mesh_new, axes=ax6)

    fig.savefig(
        os.path.join(
            problem_plot_dir, "plot_{}.png".format(i))
    )
    # fig, ax = plt.subplots()
    # ax.set_title("adapt error list")
    # ax.plot(error_adapt_list, linestyle='--', color='blue', label='adapt')
    # # ax.plot(error_og_list, linestyle='--', color='red', label='og')
    # ax.legend()
    # plt.show()

    # ==========================================
    uh = fd.project(uh, function_space_fine)
    uh_new = fd.project(uh_new, function_space_fine)

    error_original_mesh = fd.errornorm(
        uh, uh_fine, norm_type="L2"
    )
    error_optimal_mesh = fd.errornorm(
        uh_new, uh_fine, norm_type="L2"
    )
    df = pd.DataFrame({
        "error_og": error_original_mesh,
        "error_adapt": error_optimal_mesh,
        "time": dur,
    }, index=[0])
    df.to_csv(
        os.path.join(
                problem_log_dir, "log{}.csv".format(i))
        )
    print("error og/optimal:",
          error_original_mesh, error_optimal_mesh)

    i += 1
    return


# ====  Data Generation Scripts ======================
if __name__ == "__main__":
    print("In build_dataset.py")

    mesh_gen = wm.UnstructuredSquareMesh(mesh_type=mesh_type)
    mesh = mesh_gen.get_mesh(
        res=lc,
        file_path=os.path.join(problem_mesh_dir, "mesh.msh"))
    mesh_new = mesh_gen.get_mesh(
        res=lc,
        file_path=os.path.join(problem_mesh_dir, "mesh.msh"))
    mesh_gen_fine = wm.UnstructuredSquareMesh(mesh_type=mesh_type)
    mesh_fine = mesh_gen_fine.get_mesh(
        res=1e-2,
        file_path=os.path.join(problem_mesh_fine_dir, "mesh.msh"))

    # solver defination
    swril_solver = wm.SwirlSolver(
        mesh, mesh_fine, mesh_new,
        sigma=sigma, alpha=alpha, r_0=r_0,
        save_interval=save_interval,
        T=T, n_step=n_step,
    )

    swril_solver.solve_problem(
        callback=sample_from_loop,
        fail_callback=fail_callback
    )

    df = pd.DataFrame({
        'sigma': [sigma],
        'alpha': [alpha],
        'r_0': [r_0],
        'save_interval': [save_interval],
        'T': [T],
        "n_step": [n_step],
        "dt": [dt],
        "fail_t": [fail_t],
        "lc": [lc],
        "num_fail_cases": [len(fail_t)],
        "mesh_type": [mesh_type],
    })

    df.to_csv(os.path.join(problem_specific_dir, "info.csv"))

    print("Done!")
# ====  Data Generation Scripts ======================
