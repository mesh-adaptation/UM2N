# Author: Chunyang Wang
# GitHub Username: chunyang-w

import os
import pandas as pd
import warpmesh as wm
import firedrake as fd
import shutil
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--max_dist', type=int, default=6,
                        help='max number of distributions used to\
                            generate the dataset (only works if\
                                n_dist is not set)')
    parser.add_argument('--n_dist', type=int, default=None,
                        help='number of distributions used to\
                            generate the dataset (this will disable\
                                max_dist)')
    parser.add_argument('--n_grid', type=int, default=20,
                        help='number of grids of a\
                            discretized mesh')
    parser.add_argument('--field_type', type=str, default="iso",
                        help='anisotropic or isotropic data type(aniso/iso)')
    # use padded scheme or full-scale scheme to sample central point of the bump  # noqa
    parser.add_argument('--boundary_scheme', type=str, default="pad",
                        help='scheme used to generate the dataset (pad/full))')
    parser.add_argument('--n_case', type=int, default=5,
                        help='number of simulation cases')
    parser.add_argument('--rand_seed', type=int, default=63,
                        help='number of samples generated')
    args_ = parser.parse_args()
    print(args_)
    return args_


args = arg_parse()

data_type = args.field_type
use_iso = True if data_type == "iso" else False

rand_seed = args.rand_seed
random.seed(rand_seed)

# ====  Parameters ======================
problem = "burgers"

n_case = args.n_case

# parameters for domain scale
scale_x = 1
scale_y = 1

# parameters for random source
max_dist = args.max_dist
n_dist = args.n_dist
num_grid = args.n_grid
num_grid_x = num_grid
num_grid_y = num_grid

# parameters for anisotropic data - distribution height scaler
z_min = 0
z_max = 1

# parameters for isotropic data
w_min = 0.05
w_max = 0.2

scheme = args.boundary_scheme
c_min = 0.2 if scheme == "pad" else 0
c_max = 0.8 if scheme == "pad" else 1

# parameters for data split
p_train = 0.75
p_test = 0.15
p_val = 0.1

# =======================================


df = pd.DataFrame({
    'cmin': [c_min],
    'cmax': [c_max],
    'data_type': [data_type],
    'scheme': [scheme],
    'n_grid': [num_grid],
})


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
dataset_dir = os.path.join(project_dir, "data", "dataset", problem)
problem_specific_dir = os.path.join(
        dataset_dir,
        "z=<{},{}>_ndist={}_max_dist={}_<{}x{}>_n={}_{}_{}".format(
            z_min, z_max, n_dist, max_dist,
            num_grid_x, num_grid_y, n_case,
            data_type, scheme))


problem_data_dir = os.path.join(problem_specific_dir, "data")
problem_plot_dir = os.path.join(problem_specific_dir, "plot")
problem_log_dir = os.path.join(problem_specific_dir, "log")


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

df.to_csv(os.path.join(problem_specific_dir, "info.csv"))


def get_sample_param_of_nu_generalization_by_idx_train(idx_in):
    gauss_list_ = []
    if idx_in == 1:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 2:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 3:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 4:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 5:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 6:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 7:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 8:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 9:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    return gauss_list_, nu_


i = 0


def sample_from_loop(uh, uh_grad, hessian, hessian_norm,
                     phi, grad_phi,
                     jacobian, jacobian_det,
                     uh_new, mesh_og, mesh_new,
                     function_space,
                     function_space_fine,
                     uh_fine,
                     error_adapt_list,
                     error_og_list,
                     nu, gauss_list):
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
        nu=nu,
        gauss_list=gauss_list
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
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title('Original Mesh')
    fd.triplot(mesh, axes=ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title('Adapted Mesh')
    fd.triplot(mesh_new, axes=ax5)

    # plot mesh with function evaluated on it
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title('Soultion Projected on optimal mesh')
    fd.tripcolor(
        uh_new, cmap='coolwarm', axes=ax6)
    fd.triplot(mesh_new, axes=ax6)

    fig.savefig(
        os.path.join(
            problem_plot_dir, "plot_{}.png".format(i))
    )
    i += 1

    # fig, ax = plt.subplots()
    # ax.set_title("adapt error list")
    # ax.plot(error_adapt_list, linestyle='--', color='blue', label='adapt')
    # # ax.plot(error_og_list, linestyle='--', color='red', label='og')
    # ax.legend()
    # plt.show()

    # ==========================================
    uh = fd.project(uh, function_space_fine)
    uh_new = fd.project(uh_new, function_space_fine)
    # uh_fine = fd.project(uh_fine, function_space_fine)

    error_original_mesh = fd.errornorm(
        uh, uh_fine, norm_type="L2"
    )
    error_optimal_mesh = fd.errornorm(
        uh_new, uh_fine, norm_type="L2"
    )

    with open(
            os.path.join(
                problem_log_dir, "log{}.txt".format(i)), "a"
            ) as f:
        f.write(
            "error on original mesh: {}\nerror on optimal mesh: {}"
            .format(error_original_mesh, error_optimal_mesh)
        )
    print("error og/optimal:",
          error_original_mesh, error_optimal_mesh)
    return


# ====  Data Generation Scripts ======================
if __name__ == "__main__":
    print("In build_dataset.py")
    # for idx in range(1, n_case + 1):
    for idx in range(1, n_case + 1):
        try:
            print(f"Case {idx} building ...")
            mesh = fd.RectangleMesh(
                num_grid_x, num_grid_y, scale_x, scale_y)
            # Generate Random solution field
            gaussian_list, nu = get_sample_param_of_nu_generalization_by_idx_train(idx)  # noqa
            solver = wm.BurgersSolver(
                mesh, gauss_list=gaussian_list, nu=nu,
                mesh_size=num_grid
            )
            solver.solve_problem(sample_from_loop)
            print()
        except fd.exceptions.ConvergenceError:
            print("ConvergenceError")
            pass
    print("Done!")


# ====  Data Generation Scripts ======================
