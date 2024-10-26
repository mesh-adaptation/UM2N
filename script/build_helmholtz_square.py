# Author: Chunyang Wang
# GitHub Username: chunyang-w

import os
import random
import shutil
import time
from argparse import ArgumentParser

import firedrake as fd
import matplotlib.pyplot as plt
import pandas as pd

import UM2N


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--mesh_type", type=int, default=2, help="algorithm used to generate mesh"
    )
    parser.add_argument(
        "--max_dist",
        type=int,
        default=6,
        help="max number of distributions used to\
                            generate the dataset (only works if\
                                n_dist is not set)",
    )
    parser.add_argument(
        "--n_dist",
        type=int,
        default=None,
        help="number of distributions used to\
                            generate the dataset (this will disable\
                                max_dist)",
    )
    parser.add_argument(
        "--lc",
        type=float,
        default=5e-2,
        help="the length characteristic of the elements in the\
                            mesh",
    )
    parser.add_argument(
        "--field_type",
        type=str,
        default="aniso",
        help="anisotropic or isotropic data type(aniso/iso)",
    )
    # use padded scheme or full-scale scheme to sample central point of the bump  # noqa
    parser.add_argument(
        "--boundary_scheme",
        type=str,
        default="full",
        help="scheme used to generate the dataset (pad/full))",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="number of samples generated"
    )
    parser.add_argument(
        "--rand_seed", type=int, default=63, help="number of samples generated"
    )
    args_ = parser.parse_args()
    print(args_)
    return args_


args = arg_parse()

mesh_type = int(args.mesh_type)

data_type = args.field_type
use_iso = True if data_type == "iso" else False

rand_seed = args.rand_seed
random.seed(rand_seed)

# ====  Parameters ======================
problem = "holmholtz"

n_samples = args.n_samples

# parameters for domain scale
scale_x = 1
scale_y = 1

# parameters for random source
max_dist = args.max_dist
n_dist = args.n_dist
lc = args.lc

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

num_train = int(n_samples * p_train)
num_test = int(n_samples * p_test)
num_val = int(n_samples * p_val)

# parameters for dataset challenging level
sigma_mean_scaler = 1 / 4  #
sigma_sigma_scaler = (
    1 / 6
)  # larger, less challenging (because the gaussian is more like a circle)
sigma_eps = 1 / 8
# =======================================


df = pd.DataFrame(
    {
        "cmin": [c_min],
        "cmax": [c_max],
        "sigma_mean_scaler": [sigma_mean_scaler],
        "sigma_sigma_scaler": [sigma_sigma_scaler],
        "sigma_eps": [sigma_eps],
        "data_type": [data_type],
        "scheme": [scheme],
        "n_samples": [n_samples],
        "lc": [lc],
        "mesh_type": [mesh_type],
    }
)


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
            os.path.join(source, f"data_{i:04d}.npy"),
            os.path.join(target, f"data_{i:04d}.npy"),
        )


project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
dataset_dir = os.path.join(
    project_dir, "data", f"dataset_meshtype_{mesh_type}", "helmholtz"
)  # noqa
problem_specific_dir = os.path.join(
    dataset_dir,
    "z=<{},{}>_ndist={}_max_dist={}_lc={}_n={}_{}_{}_meshtype_{}".format(
        z_min, z_max, n_dist, max_dist, lc, n_samples, data_type, scheme, mesh_type
    ),
)


problem_data_dir = os.path.join(problem_specific_dir, "data")
problem_plot_dir = os.path.join(problem_specific_dir, "plot")
problem_plot_compare_dir = os.path.join(problem_specific_dir, "plot_compare")
problem_log_dir = os.path.join(problem_specific_dir, "log")

problem_mesh_dir = os.path.join(problem_specific_dir, "mesh")
problem_mesh_fine_dir = os.path.join(problem_specific_dir, "mesh_fine")
problem_train_dir = os.path.join(problem_specific_dir, "train")
problem_test_dir = os.path.join(problem_specific_dir, "test")
problem_val_dir = os.path.join(problem_specific_dir, "val")

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

df.to_csv(os.path.join(problem_specific_dir, "info.csv"))


# ====  Data Generation Scripts ======================
if __name__ == "__main__":
    print("In build_dataset.py")
    i = 0
    while i < n_samples:
        try:
            print("Generating Sample: " + str(i))
            if mesh_type != 0:
                unstructure_square_mesh_gen = UM2N.UnstructuredSquareMesh(
                    scale=scale_x, mesh_type=mesh_type
                )  # noqa
                mesh = unstructure_square_mesh_gen.get_mesh(
                    res=lc,
                    file_path=os.path.join(problem_mesh_dir, f"mesh_{i:04d}.msh"),
                )
            else:
                n_grid = int(1 / lc)
                mesh = fd.UnitSquareMesh(n_grid, n_grid)

            # Generate Random solution field
            rand_u_generator = UM2N.RandSourceGenerator(
                use_iso=use_iso,
                dist_params={
                    "max_dist": max_dist,
                    "n_dist": n_dist,
                    "x_start": 0,
                    "x_end": 1,
                    "y_start": 0,
                    "y_end": 1,
                    "z_max": z_max,
                    "z_min": z_min,
                    "w_min": w_min,
                    "w_max": w_max,
                    "c_min": c_min,
                    "c_max": c_max,
                    "sigma_mean_scaler": sigma_mean_scaler,
                    "sigma_sigma_scaler": sigma_sigma_scaler,
                    "sigma_eps": sigma_eps,
                },
            )
            helmholtz_eq = UM2N.RandHelmholtzEqGenerator(rand_u_generator)
            res = helmholtz_eq.discretise(mesh)  # discretise the equation
            dist_params = rand_u_generator.get_dist_params()
            # Solve the equation
            solver = UM2N.EquationSolver(
                params={
                    "function_space": res["function_space"],
                    "LHS": res["LHS"],
                    "RHS": res["RHS"],
                    "bc": res["bc"],
                }
            )
            # RHS of helmholtz problem
            f = fd.interpolate(helmholtz_eq.f, helmholtz_eq.function_space)
            # fd.trisurf(f)
            # plt.show()
            uh = solver.solve_eq()
            # Generate Mesh
            hessian = UM2N.MeshGenerator(
                params={"eq": helmholtz_eq, "mesh": mesh}
            ).get_hessian(mesh)

            hessian_norm = UM2N.MeshGenerator(
                params={"eq": helmholtz_eq, "mesh": mesh}
            ).get_hessian_norm(mesh)
            hessian_norm = fd.project(hessian_norm, fd.FunctionSpace(mesh, "CG", 1))

            # Get monitor val
            monitor_val = UM2N.MeshGenerator(
                params={"eq": helmholtz_eq, "mesh": mesh}
            ).monitor_func(mesh)

            # grad_uh_norm = UM2N.MeshGenerator(
            #     params={
            #         "eq": helmholtz_eq,
            #         "mesh": fd.Mesh(
            #             os.path.join(problem_mesh_dir, f"mesh_{i:04d}.msh")
            #         ),  # noqa
            #     }
            # ).get_grad_norm(mesh)

            func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
            grad_uh_interpolate = fd.interpolate(fd.grad(uh), func_vec_space)

            grad_norm = fd.Function(res["function_space"])
            grad_norm.project(grad_uh_interpolate[0] ** 2 + grad_uh_interpolate[1] ** 2)
            grad_norm /= grad_norm.vector().max()
            grad_uh_norm = grad_norm

            mesh_gen = UM2N.MeshGenerator(params={"eq": helmholtz_eq, "mesh": mesh})

            start = time.perf_counter()
            new_mesh = mesh_gen.move_mesh()  # noqa
            end = time.perf_counter()
            dur = (end - start) * 1000

            # Get monitor val
            # monitor_val = mesh_gen.get_monitor_val()

            # this is the jacobian of x with respect to xi
            jacobian = mesh_gen.get_jacobian()
            jacobian = fd.project(jacobian, fd.TensorFunctionSpace(new_mesh, "CG", 1))
            jacobian_det = mesh_gen.get_jacobian_det()
            jacobian_det = fd.project(jacobian_det, fd.FunctionSpace(new_mesh, "CG", 1))

            # get phi/grad_phi projected to the original mesh
            phi = mesh_gen.get_phi()
            # phi = fd.project(
            #     phi, fd.FunctionSpace(mesh, "CG", 1)
            # )
            grad_phi = mesh_gen.get_grad_phi()
            # grad_phi = fd.project(
            #     grad_phi, fd.VectorFunctionSpace(mesh, "CG", 1)
            # )

            # solve the equation on the new mesh
            new_res = helmholtz_eq.discretise(new_mesh)
            new_solver = UM2N.EquationSolver(
                params={
                    "function_space": new_res["function_space"],
                    "LHS": new_res["LHS"],
                    "RHS": new_res["RHS"],
                    "bc": new_res["bc"],
                }
            )
            uh_new = new_solver.solve_eq()

            # process the data for training
            mesh_processor = UM2N.MeshProcessor(
                original_mesh=mesh,
                optimal_mesh=new_mesh,
                function_space=new_res["function_space"],
                use_4_edge=True,
                feature={
                    "uh": uh.dat.data_ro.reshape(-1, 1),
                    "grad_uh": grad_uh_interpolate.dat.data_ro.reshape(-1, 2),
                    "grad_uh_norm": grad_uh_norm.dat.data_ro.reshape(-1, 1),
                    "hessian": hessian.dat.data_ro.reshape(-1, 4),
                    "hessian_norm": hessian_norm.dat.data_ro.reshape(-1, 1),
                    "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
                    "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
                    "phi": phi.dat.data_ro.reshape(-1, 1),
                    "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
                    "f": f.dat.data_ro.reshape(-1, 1),
                    "monitor_val": monitor_val.dat.data_ro.reshape(-1, 1),
                },
                raw_feature={
                    "uh": uh,
                    "hessian_norm": hessian_norm,
                    "monitor_val": monitor_val,
                    "jacobian": jacobian,
                    "jacobian_det": jacobian_det,
                },
                dist_params=dist_params,
            )

            mesh_processor.save_taining_data(
                os.path.join(problem_data_dir, f"data_{i:04d}")
            )

            # # ====  Plot Scripts ======================
            # fig = plt.figure(figsize=(15, 10))
            # ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            # # Plot the exact solution
            # ax1.set_title('Exact Solution')
            # fd.trisurf(fd.interpolate(
            #     res["u_exact"], res["function_space"]), axes=ax1)
            # # Plot the solved solution
            # ax2 = fig.add_subplot(2, 3, 2, projection='3d')
            # ax2.set_title('FEM Solution')
            # fd.trisurf(uh, axes=ax2)

            # # Plot the solution on a optimal mesh
            # ax3 = fig.add_subplot(2, 3, 3, projection='3d')
            # ax3.set_title('FEM Solution on Optimal Mesh')
            # fd.trisurf(uh_new, axes=ax3)

            # # Plot the mesh
            # ax4 = fig.add_subplot(2, 3, 4)
            # ax4.set_title('Original Mesh')
            # fd.triplot(mesh, axes=ax4)
            # ax5 = fig.add_subplot(2, 3, 5)
            # ax5.set_title('Optimal Mesh')
            # fd.triplot(new_mesh, axes=ax5)

            # # plot mesh with function evaluated on it
            # ax6 = fig.add_subplot(2, 3, 6)
            # ax6.set_title('Soultion Projected on optimal mesh')
            # fd.tripcolor(
            #     uh_new, cmap='coolwarm', axes=ax6)
            # fd.triplot(new_mesh, axes=ax6)

            # fig.savefig(
            #     os.path.join(
            #         problem_plot_dir, f"plot_{i:04d}.png")
            # )

            # ==========================================

            if mesh_type != 0:
                # generate log file
                high_res_mesh = unstructure_square_mesh_gen.get_mesh(
                    res=1e-2,
                    file_path=os.path.join(problem_mesh_fine_dir, f"mesh_{i:04d}.msh"),
                )
            else:
                high_res_mesh = fd.UnitSquareMesh(100, 100)
            high_res_function_space = fd.FunctionSpace(high_res_mesh, "CG", 1)

            res_high_res = helmholtz_eq.discretise(high_res_mesh)
            u_exact = fd.interpolate(
                res_high_res["u_exact"], res_high_res["function_space"]
            )

            uh_proj = fd.project(uh, high_res_function_space)
            uh_new_proj = fd.project(uh_new, high_res_function_space)

            error_original_mesh = fd.errornorm(u_exact, uh_proj)
            error_optimal_mesh = fd.errornorm(u_exact, uh_new_proj)

            df = pd.DataFrame(
                {
                    "error_og": error_original_mesh,
                    "error_adapt": error_optimal_mesh,
                    "time": dur,
                },
                index=[0],
            )
            df.to_csv(os.path.join(problem_log_dir, f"log_{i:04d}.csv"))
            print("error og/optimal:", error_original_mesh, error_optimal_mesh)

            # ====  Plot mesh, solution, error ======================
            rows, cols = 3, 3
            fig, ax = plt.subplots(
                rows, cols, figsize=(cols * 5, rows * 5), layout="compressed"
            )

            # High resolution mesh
            fd.triplot(high_res_mesh, axes=ax[0, 0])
            ax[0, 0].set_title("High resolution Mesh ")
            # Orginal low resolution uniform mesh
            fd.triplot(mesh, axes=ax[0, 1])
            ax[0, 1].set_title("Original uniform Mesh")
            # Adapted mesh
            fd.triplot(new_mesh, axes=ax[0, 2])
            ax[0, 2].set_title("Adapted Mesh (MA)")

            cmap = "seismic"
            # Solution on high resolution mesh
            cb = fd.tripcolor(u_exact, cmap=cmap, axes=ax[1, 0])
            ax[1, 0].set_title("Solution on High Resolution (u_exact)")
            plt.colorbar(cb)
            # Solution on orginal low resolution uniform mesh
            cb = fd.tripcolor(uh, cmap=cmap, axes=ax[1, 1])
            ax[1, 1].set_title("Solution on uniform Mesh")
            plt.colorbar(cb)
            # Solution on adapted mesh
            cb = fd.tripcolor(uh_new, cmap=cmap, axes=ax[1, 2])
            ax[1, 2].set_title("Solution on Adapted Mesh (MA)")
            plt.colorbar(cb)

            err_orignal_mesh = fd.assemble(uh_proj - u_exact)
            err_adapted_mesh = fd.assemble(uh_new_proj - u_exact)
            err_abs_max_val_ori = max(
                abs(err_orignal_mesh.dat.data[:].max()),
                abs(err_orignal_mesh.dat.data[:].min()),
            )
            err_abs_max_val_adapted = max(
                abs(err_adapted_mesh.dat.data[:].max()),
                abs(err_adapted_mesh.dat.data[:].min()),
            )
            err_abs_max_val = max(err_abs_max_val_ori, err_abs_max_val_adapted)
            err_v_max = err_abs_max_val
            err_v_min = -err_v_max

            # Error on high resolution mesh
            cb = fd.tripcolor(monitor_val, cmap=cmap, axes=ax[2, 0])
            ax[2, 0].set_title("Monitor values")
            plt.colorbar(cb)
            # Error on orginal low resolution uniform mesh
            cb = fd.tripcolor(
                err_orignal_mesh,
                cmap=cmap,
                axes=ax[2, 1],
                vmax=err_v_max,
                vmin=err_v_min,
            )
            ax[2, 1].set_title(
                f"Error (u-u_exact) uniform Mesh | L2 Norm: {error_original_mesh:.5f}"
            )
            plt.colorbar(cb)
            # Error on adapted mesh
            cb = fd.tripcolor(
                err_adapted_mesh,
                cmap=cmap,
                axes=ax[2, 2],
                vmax=err_v_max,
                vmin=err_v_min,
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
        except fd.exceptions.ConvergenceError:
            print(f"Iteration: {i}, not coverged.")
            pass
        # except AttributeError:
        #     print(f"AttributeError")
        #     pass
        # except ValueError:
        #     pass

    move_data(problem_train_dir, problem_data_dir, 0, num_train)

    move_data(problem_test_dir, problem_data_dir, num_train, num_train + num_test)

    move_data(
        problem_val_dir,
        problem_data_dir,
        num_train + num_test,
        num_train + num_test + num_val,
    )
# ====  Data Generation Scripts ======================
