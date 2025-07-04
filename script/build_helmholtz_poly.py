# Author: Chunyang Wang
# GitHub Username: chunyang-w
import csv
import os
import random
import shutil
import time
from argparse import ArgumentParser

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from firedrake.__future__ import interpolate

# dd the parent directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import UM2N

def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Build Burgers dataset with square meshes.")
    parser.add_argument("--mesh_type", type=int, default=2, help="Algorithm used to generate mesh.")
    parser.add_argument("--max_dist", type=int, default=6, help="Max number of distributions.")
    parser.add_argument("--n_dist", type=int, default=None, help="Number of distributions.")
    parser.add_argument("--lc", type=float, default=6e-2, help="Length characteristic of mesh elements.")
    parser.add_argument("--field_type", type=str, default="iso", help="Data type (aniso/iso).")
    # use padded scheme or full-scale scheme to sample central point of the bump  # noqa
    parser.add_argument("--boundary_scheme", type=str, default="pad", help="Boundary scheme (pad/full).")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples generated")
    parser.add_argument("--rand_seed", type=int, default=63, help="Random seed")
    
    parsed_args = parser.parse_args()

    # Handle dependency between max_dist and n_dist
    # max number of distributions used to generate the dataset
    # only if n_dist is not set if n_dist is set, max_dist will be disabled
    if parsed_args.n_dist is not None:
        parsed_args.max_dist = None  # Disable max_dist if n_dist is set
        print("Warning: max_dist is ignored because n_dist is set.")
    # QC:
    print(parsed_args)
    
    return parser.parse_args()

def setup_directories(problem, mesh_type, base_dir= None, subdirs=None, dir_format=None):
    """
    Set up directories for storing data, plots, and logs.

    Args:
        base_dir (str): Base directory for the project.
        parameters (dict): Dictionary of parameters, including "mesh_type" and "problem".
            - "mesh_type" (int): Type of mesh used in the simulation (default: 0).
            - "problem" (str): Name of the problem (e.g., "burgers" or "helmholtz") (default: "default_problem").
        subdirs (list, optional): List of subdirectories to create. Defaults to:
            ["data", "plot", "log", "mesh", "mesh_fine"].
            Additional subdirectories like "plot_compare", "train", "test", and "val" are added for "helmholtz".
        dir_format (str, optional): Format string for the problem-specific directory. Must use placeholders
            matching keys in the `parameters` dictionary. Example:
            "lc={lc}_ngrid_{n_grid}_n={n_case}_{data_type}_{scheme}_meshtype_{mesh_type}".
            If not provided, raises a ValueError.

    Returns:
        dict: A dictionary mapping subdirectory names to their full paths.

    Raises:
        ValueError: If `dir_format` is not provided or is invalid.
    """

    # Define the project directory
    if base_dir:
        project_dir = os.path.abspath(base_dir)
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # QC:
    print(f"Project Directory: {project_dir}")

    # Define the dataset directory
    dataset_dir = os.path.join(project_dir, "data", f"dataset_meshtype_{mesh_type}", problem)

    # Use the provided format string for the problem-specific directory
    if dir_format is None:
        problem_specific_dir = os.path.join(dataset_dir, f"{problem}_meshtype_{mesh_type}")
    else:
        # check if dir_format is a valid string format
        if not isinstance(dir_format, str):
            raise ValueError("dir_format must be a string.")
        problem_specific_dir = os.path.join(dataset_dir, dir_format)

    # Define default subdirectories if not provided
    if subdirs is None:
        subdirs = ["data", "plot", "log", "mesh", "mesh_fine",
                   "plot_compare", "train", "test", "val"]

    # Create and clear directories
    directories = {}
    for subdir in subdirs:
        dir_path = os.path.join(problem_specific_dir, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            # Clear the directory by removing all files
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
        directories[subdir] = dir_path

    # QC:
    print(f"Subdirectories created: {directories}")

    return directories

def output_csv(parameters, key_list, output_dir):
    """
    Write selected parameters to a CSV file.

    Args:
        parameters (dict): Dictionary of parameters to write.
        key_list (list): List of keys to include in the CSV.
        output_dir (str): Directory where the CSV file will be saved.
    """
    # Filter parameters based on key_list
    csv_keys = [key for key in key_list if key in parameters]
    csv_data = [parameters[key] for key in csv_keys]

    # Define the output file path
    csv_file_path = os.path.join(output_dir, "info.csv")

    # Write to CSV
    with open(csv_file_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header (keys)
        csv_writer.writerow(csv_keys)
        # Write data (values)
        csv_writer.writerow(csv_data)

def move_data(target, source, start, num_files):
    """
    Move data files from the source directory to the target directory.

    Args:
        target (str): The path to the target directory.
        source (str): The path to the source directory.
        start (int): The starting index of the files to move.
        num_files (int): The total number of files to move.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        ValueError: If the start index or num_files is invalid.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source directory '{source}' does not exist.")

    if start < 0 or num_files <= 0:
        raise ValueError("Invalid start index or number of files to move.")

    # Create the target directory if it doesn't exist
    if not os.path.exists(target):
        os.makedirs(target)
    else:
        # Clear the target directory by removing all files
        for file in os.listdir(target):
            os.remove(os.path.join(target, file))

    # Copy files sequentially starting from the specified index
    for i in range(start, start + num_files):
        try:
            # Copy the data file
            shutil.copy(
                os.path.join(source, f"data_{i:04d}.npy"),
                os.path.join(target, f"data_{i:04d}.npy"),
            )
        except FileNotFoundError:
            print(f"File data_{i:04d}.npy not found in {source}. Skipping.")
            continue
        except Exception as e:
            print(f"An error occurred while copying data_{i:04d}.npy: {e}")
            continue

def process_features(parameters, dirs):

    mesh_type = parameters["mesh_type"]
    scale_x = parameters["scale_x"]
    lc = parameters["lc"]

    # create mesh
    rand_poly_mesh_gen = UM2N.UnstructuredRandomPolygonalMeshGenerator(
        scale=scale_x, mesh_type=mesh_type
    )  # noqa
    mesh = rand_poly_mesh_gen.generate_mesh(
        res=lc, output_filename=os.path.join(dirs["mesh"], f"mesh{i}.msh")
    )
    num_boundary = rand_poly_mesh_gen.num_boundary

    # Generate Random solution field
    rand_u_generator = UM2N.RandSourceGenerator(
        use_iso= parameters["data_type"] == "iso",
        dist_params= parameters
    )

    # generate equation
    helmholtz_eq = UM2N.RandHelmholtzEqGenerator(rand_u_generator)
    # discretise the equation
    res = helmholtz_eq.discretise(mesh)
    # get specific parameters used
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

    # original solution field
    uh = solver.solve_eq()

    func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    grad_uh_interpolate = fd.assemble(interpolate(fd.grad(uh), func_vec_space))

    # ej321 - grad_norm copied from build_helmholtz_square.py
    grad_norm = fd.Function(res["function_space"])
    grad_norm.project(grad_uh_interpolate[0] ** 2 + grad_uh_interpolate[1] ** 2)
    grad_norm /= grad_norm.vector().max()

    # RHS of helmholtz problem
    f_rhs = fd.assemble(interpolate(helmholtz_eq.f, helmholtz_eq.function_space))

    # ej321 - this seems extra - the mesh is never used, just to build 'eq'?
    # hessian = UM2N.MeshGenerator(
    #     params={
    #         "eq": helmholtz_eq,
    #         "mesh": rand_poly_mesh_gen.generate_mesh(
    #             res=lc,
    #             output_filename=os.path.join(problem_mesh_dir, f"mesh{i}.msh"),
    #         ),
    #     }
    # ).get_hessian(mesh)
    # ej321 - using script from build_helmholtz_square.py
    mesh_gen = UM2N.MeshGenerator(params={"eq": helmholtz_eq, "mesh": mesh})
    monitor_val = mesh_gen.monitor_func(mesh)
    hessian = mesh_gen.get_hessian(mesh)
    hessian_norm = fd.project(mesh_gen.get_hessian_norm(mesh),
                                fd.FunctionSpace(mesh, "CG", 1)
                                )


    # move the mesh?
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
    grad_phi = mesh_gen.get_grad_phi()

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
        use_4_edge=False,
        num_boundary=num_boundary,
        feature={
            "uh": uh.dat.data_ro.reshape(-1, 1),
            "grad_uh": grad_uh_interpolate.dat.data_ro.reshape(-1, 2),
            "grad_uh_norm": grad_norm.dat.data_ro.reshape(-1, 1), #ej321 - added
            "hessian": hessian.dat.data_ro.reshape(-1, 4),
            "hessian_norm": hessian_norm.dat.data_ro.reshape(-1, 1),
            "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
            "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
            "phi": phi.dat.data_ro.reshape(-1, 1),
            "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
            "f": f_rhs.dat.data_ro.reshape(-1, 1),
            "monitor_val": monitor_val.dat.data_ro.reshape(-1, 1), # ej321 - added
        },
        raw_feature={
            "uh": uh,
            "hessian_norm": hessian_norm,
            "monitor_val": monitor_val, # ej321 - added
            "grad_uh_norm": grad_norm, # ej321 - added needed for poly only
            "jacobian": jacobian,
            "jacobian_det": jacobian_det,
        },
        dist_params=dist_params,
        poly_mesh=True,
    )

    # save out data
    mesh_processor.save_taining_data(
        os.path.join(dirs["data"], "data_{}".format(i))
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

    fig.savefig(os.path.join(dirs["plot"], "plot_{}.png".format(i)))


    # ====  Log File ============================================
    high_res_mesh = rand_poly_mesh_gen.generate_mesh(
        res=1e-2,
        output_filename=os.path.join(dirs["mesh_fine"], f"mesh{i}.msh"),
    )

    high_res_function_space = fd.FunctionSpace(high_res_mesh, "CG", 1)

    res_high_res = helmholtz_eq.discretise(high_res_mesh)
    u_exact = fd.assemble(interpolate(res_high_res["u_exact"],
                    res_high_res["function_space"])
                    )

    uh_proj = fd.project(uh, high_res_function_space)
    uh_new_proj = fd.project(uh_new, high_res_function_space)

    error_original_mesh = fd.errornorm(u_exact, uh_proj)
    error_optimal_mesh = fd.errornorm(u_exact, uh_new_proj)

    # df = pd.DataFrame(
    #     {
    #         "error_og": error_original_mesh,
    #         "error_adapt": error_optimal_mesh,
    #         "time": dur,
    #     },
    #     index=[0],
    # )
    # df.to_csv(os.path.join(problem_log_dir, "log{}.csv".format(i)))

    # Write to CSV
    with open(os.path.join(dirs["log"], f"log_{i:04d}.csv"), mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header (keys)
        csv_writer.writerow(["error_og", "error_adapt", "time"])
        # Write data (values)
        csv_writer.writerow([error_original_mesh, error_optimal_mesh, dur])
    print("error og/optimal:", error_original_mesh, error_optimal_mesh)



if __name__ == "__main__":

    # parse args
    args = parse_arguments()
    
    # ====  Parameters ======================
    parameters = {
        # parameters for problem
        "problem": "holmholtz_poly",
        # "n_case": args.n_case, # burgers problem only
        # parameters for random source
        "n_dist": args.n_dist,
        "max_dist": args.max_dist,
        "lc": args.lc,
        # "n_grig": args.n_grid, # burgers problem only
        # parameters for ??????
        "n_samples": args.n_samples,
        "data_type": args.field_type,
        "scheme": args.boundary_scheme,
        "mesh_type": int(args.mesh_type),
        # parameters for domain scale
        "scale_x": 1,
        "scale_y": 1,
        # parameters for anisotropic data - distribution height scaler
        "z_max": 1,
        "z_min": 0,
        # parameters for ?????
        "x_start": 0,
        "x_end": 1,
        "y_start": 0,
        "y_end": 1,
        # parameters for isotropic data
        "w_min": 0.05,
        "w_max": 0.2,
        "c_min": 0.3 if args.boundary_scheme == "pad" else 0,
        "c_max": 0.7 if args.boundary_scheme == "pad" else 1,
        # parameters for dataset challenging level
        # larger, less challenging (because the gaussian is more like a circle)
        "sigma_mean_scaler": 1 / 4,
        "sigma_sigma_scaler": 1 / 6,
        "sigma_eps": 1 / 8,
        # parameters for data split
        "p_train": 0.75,
        "p_test": 0.15,
        "p_val": 0.1,
    }

    # Set random seed
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    # ====  Setup Directories ======================
    problem_specific_dir = "z=<{},{}>_ndist={}_max_dist={}_lc={}_n={}_{}_{}_meshtype_{}".format(
            parameters["z_min"], parameters["z_max"],
            parameters["n_dist"],parameters["max_dist"],
            parameters["lc"], parameters["n_samples"],
            parameters["data_type"], parameters["scheme"], parameters["mesh_type"]
        )

    subdirs = [
        "data", "plot", "plot_compare", "log", "mesh", "mesh_fine",
        "train", "test", "val",
    ]

    dirs = setup_directories(problem = parameters["problem"],
                        mesh_type = parameters["mesh_type"],
                        base_dir = None,
                        subdirs = subdirs,
                        dir_format = problem_specific_dir)


    # ====  Output CSV ======================
    key_list = [
        "cmin","cmax",
        "data_type", "scheme", "n_samples", "lc", "mesh_type"
    ]
    output_csv(parameters, key_list, dirs["data"])

    # ====  Data Generation Scripts ======================
    # QC: 
    print("In build_dataset.py")
    # i = 0
    # while i < n_samples:
    for i in range(parameters["n_samples"]):
        try:
            print("Generating Sample: " + str(i))
          
            process_features(parameters, dirs)
            # i += 1
        except fd.exceptions.ConvergenceError:
            print(f"Iteration {i} did not converge.")
            continue
        except AttributeError:
            pass
        except ValueError:
            pass

    # ====  Data Splits ============================================
    num_train = int(parameters["n_samples"] * parameters["p_train"])
    num_test = int(parameters["n_samples"] * parameters["p_test"])
    num_val = parameters["n_samples"] - num_train - num_test

    move_data(dirs["train"], dirs["data"], 0, num_train)
    move_data(dirs["test"], dirs["data"], num_train, num_train + num_test)
    move_data(dirs["val"], dirs["data"], num_train + num_test, num_train + num_test + num_val)
