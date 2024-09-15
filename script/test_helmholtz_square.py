# Author: Chunyang Wang
# GitHub Username: chunyang-w

import os
import random
import shutil
import time
from argparse import ArgumentParser
from types import SimpleNamespace

import firedrake as fd
import matplotlib.pyplot as plt
import pandas as pd
import wandb

import warpmesh as wm

entity = "w-chunyang"
project_name = "warpmesh"
run_id = "sr7waaso"
epoch = 1299


def arg_parse():
    parser = ArgumentParser()
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
        "--n_grid",
        type=int,
        default=20,
        help="number of grids of a\
                            discretized mesh",
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
        "--n_samples", type=int, default=3, help="number of samples generated"
    )
    parser.add_argument(
        "--rand_seed", type=int, default=63, help="number of samples generated"
    )
    args_ = parser.parse_args()
    print(args_)
    return args_


args = arg_parse()

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

num_train = int(n_samples * p_train)
num_test = int(n_samples * p_test)
num_val = int(n_samples * p_val)
# =======================================


df = pd.DataFrame(
    {
        "cmin": [c_min],
        "cmax": [c_max],
        "data_type": [data_type],
        "scheme": [scheme],
        "n_samples": [n_samples],
        "n_grid": [num_grid],
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
            os.path.join(source, "data_{}.npy".format(i)),
            os.path.join(target, "data_{}.npy".format(i)),
        )


project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
dataset_dir = os.path.join(project_dir, "data", "dataset", "helmholtz")
problem_specific_dir = os.path.join(
    dataset_dir,
    "z=<{},{}>_ndist={}_max_dist={}_<{}x{}>_n={}_{}_{}".format(
        z_min,
        z_max,
        n_dist,
        max_dist,
        num_grid_x,
        num_grid_y,
        n_samples,
        data_type,
        scheme,
    ),
)


problem_data_dir = os.path.join(problem_specific_dir, "data")
problem_plot_dir = os.path.join(problem_specific_dir, "plot")
problem_log_dir = os.path.join(problem_specific_dir, "log")

problem_train_dir = os.path.join(problem_specific_dir, "train")
problem_test_dir = os.path.join(problem_specific_dir, "test")
problem_val_dir = os.path.join(problem_specific_dir, "val")


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


# ====  Data Generation Scripts ======================
if __name__ == "__main__":
    print("In build_dataset.py")
    i = 0
    while i < n_samples:
        try:
            print("Generating Sample: " + str(i))
            mesh = fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y)
            # Generate Random solution field
            rand_u_generator = wm.RandSourceGenerator(
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
                },
            )
            helmholtz_eq = wm.RandHelmholtzEqGenerator(rand_u_generator)
            res = helmholtz_eq.discretise(mesh)  # discretise the equation
            dist_params = rand_u_generator.get_dist_params()
            print("dist_params", dist_params)
            # Solve the equation
            solver = wm.EquationSolver(
                params={
                    "function_space": res["function_space"],
                    "LHS": res["LHS"],
                    "RHS": res["RHS"],
                    "bc": res["bc"],
                }
            )
            uh = solver.solve_eq()
            # Generate Mesh
            hessian = wm.MeshGenerator(
                params={
                    "num_grid_x": num_grid_x,
                    "num_grid_y": num_grid_y,
                    "eq": helmholtz_eq,
                    "mesh": fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y),
                }
            ).get_hessian(mesh)

            hessian_norm = wm.MeshGenerator(
                params={
                    "num_grid_x": num_grid_x,
                    "num_grid_y": num_grid_y,
                    "eq": helmholtz_eq,
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
                    "eq": helmholtz_eq,
                    "mesh": fd.RectangleMesh(num_grid_x, num_grid_y, scale_x, scale_y),
                }
            )

            start = time.perf_counter()
            new_mesh = mesh_gen.move_mesh()
            end = time.perf_counter()
            dur = (end - start) * 1000

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
            new_solver = wm.EquationSolver(
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
                use_4_edge=True,
                feature={
                    "uh": uh.dat.data_ro.reshape(-1, 1),
                    "grad_uh": grad_uh_interpolate.dat.data_ro.reshape(-1, 2),
                    "hessian": hessian.dat.data_ro.reshape(-1, 4),
                    "hessian_norm": hessian_norm.dat.data_ro.reshape(-1, 1),
                    "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
                    "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
                    "phi": phi.dat.data_ro.reshape(-1, 1),
                    "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
                },
                raw_feature={
                    "uh": uh,
                    "hessian_norm": hessian_norm,
                    "jacobian": jacobian,
                    "jacobian_det": jacobian_det,
                },
                dist_params=dist_params,
            )

            mesh_processor.save_taining_data(
                os.path.join(problem_data_dir, "data_{}".format(i))
            )

            # ====  Plot Scripts ======================
            fig = plt.figure(figsize=(15, 10))
            ax1 = fig.add_subplot(2, 3, 1, projection="3d")
            # Plot the exact solution
            ax1.set_title("Exact Solution")
            fd.trisurf(fd.interpolate(res["u_exact"], res["function_space"]), axes=ax1)
            # Plot the solved solution
            ax2 = fig.add_subplot(2, 3, 2, projection="3d")
            ax2.set_title("FEM Solution")
            fd.trisurf(uh, axes=ax2)

            # Plot the solution on a optimal mesh
            ax3 = fig.add_subplot(2, 3, 3, projection="3d")
            ax3.set_title("FEM Solution on Optimal Mesh")
            fd.trisurf(uh_new, axes=ax3)

            # Plot the mesh
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.set_title("Original Mesh")
            fd.triplot(mesh, axes=ax4)
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.set_title("Optimal Mesh")
            fd.triplot(new_mesh, axes=ax5)

            # plot mesh with function evaluated on it
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.set_title("Soultion Projected on optimal mesh")
            fd.tripcolor(uh_new, cmap="coolwarm", axes=ax6)
            fd.triplot(new_mesh, axes=ax6)

            fig.savefig(os.path.join(problem_plot_dir, "plot_{}.png".format(i)))

            # ==========================================

            # generate log file
            high_res_mesh = fd.UnitSquareMesh(80, 80)
            high_res_function_space = fd.FunctionSpace(high_res_mesh, "CG", 1)

            res_high_res = helmholtz_eq.discretise(high_res_mesh)
            # u_exact = res_high_res["u_exact"]
            u_exact = fd.interpolate(
                res_high_res["u_exact"], res_high_res["function_space"]
            )

            uh = fd.project(uh, high_res_function_space)
            uh_new = fd.project(uh_new, high_res_function_space)

            error_original_mesh = fd.errornorm(u_exact, uh)
            error_optimal_mesh = fd.errornorm(u_exact, uh_new)

            api = wandb.Api()
            run = api.run(f"{entity}/{project_name}/{run_id}")
            config = SimpleNamespace(**run.config)
            dataset_path = problem_data_dir
            dataset = wm.MeshDataset(
                dataset_path,
                transform=wm.normalise if wm.normalise else None,
                x_feature=config.x_feat,
                mesh_feature=config.mesh_feat,
                conv_feature=config.conv_feat,
                conv_feature_fix=config.conv_feat_fix,
                load_analytical=True,
                use_run_time_cluster=False,
                use_cluster=False,
                r=0.35,
                M=25,
                dist_weight=False,
                add_nei=True,
            )
            print(dataset[i].dist_params)
            compare_res = wm.compare_error(dataset[i], mesh, high_res_mesh, new_mesh, 0)

            print("compare_res", compare_res)

            mesh_dataset = fd.UnitSquareMesh(num_grid_x, num_grid_y)
            mesh_dataset.coordinates.dat.data[:] = dataset[i].y.detach().cpu().numpy()  # noqa

            # fig = plt.figure(figsize=(10, 5))

            # ax1 = fig.add_subplot(2, 1, 1, projection='3d')
            # fd.trisurf(compare_res["u_exact"], axes=ax1)
            # ax1.set_title("u_exact from compare function")

            # ax2 = fig.add_subplot(2, 1, 2, projection='3d')
            # fd.trisurf(u_exact, axes=ax2)
            # ax2.set_title("u_exact og")

            df = pd.DataFrame(
                {
                    "error_og": error_original_mesh,
                    "error_adapt": error_optimal_mesh,
                    "time": dur,
                },
                index=[0],
            )
            df.to_csv(os.path.join(problem_log_dir, "log{}.csv".format(i)))
            print("error og/optimal:", error_original_mesh, error_optimal_mesh)
            i += 1
            plt.show()
        except fd.exceptions.ConvergenceError:
            pass

    move_data(problem_train_dir, problem_data_dir, 0, num_train)

    move_data(problem_test_dir, problem_data_dir, num_train, num_train + num_test)

    move_data(
        problem_val_dir,
        problem_data_dir,
        num_train + num_test,
        num_train + num_test + num_val,
    )
# ====  Data Generation Scripts ======================
