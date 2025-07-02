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
from firedrake.__future__ import interpolate

import UM2N


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--mesh_type", type=int, default=2, help="Algorithm used to generate mesh")
    parser.add_argument("--max_dist", type=int, default=6, help="Max number of distributions")
    parser.add_argument("--n_dist", type=int, default=None, help="Number of distributions")
    parser.add_argument("--lc", type=float, default=5e-2, help="Length characteristic of mesh elements")
    parser.add_argument("--field_type", type=str, default="aniso", help="Data type (aniso/iso)")
    # use padded scheme or full-scale scheme to sample central point of the bump  # noqa
    parser.add_argument("--boundary_scheme", type=str, default="full", help="Boundary scheme (pad/full)")
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

    return parsed_args


def setup_and_clear_directories(base_dir, subdirs):
    """
    Create and clear multiple directories under a base directory.

    Args:
        base_dir: The base directory where subdirectories will be created.
        subdirs: A list of subdirectory names to create and clear.

    Returns:
        dict: A dictionary mapping subdirectory names to their full paths.
    """
    paths = {}
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            # Clear the directory by removing all files
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
        paths[subdir] = path
    return paths


def move_data(target, source, start, num_files):
    """
    Move data files from the source directory to the target directory.

    Args:
        target: The path to the target directory.
        source: The path to the source directory.
        start: The starting index of the files to move.
        num_files: The total number of files to move.

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

    #copy files sequentially starting from the specified index
    for i in range(start, num_files):
        shutil.copy(
            os.path.join(source, f"data_{i:04d}.npy"),
            os.path.join(target, f"data_{i:04d}.npy"),
        )



def create_mesh(i, mesh_type, lc, scale_x, problem_mesh_dir):
    """
    Generate a mesh for the given sample index.

    Args:
        i: The sample index.
        mesh_type: The type of mesh to generate.
        lc: The length characteristic of the mesh.
        scale_x: The scale of the mesh.
        problem_mesh_dir: Directory to save the generated mesh.

    Returns:
        The generated mesh.
    """
    if mesh_type != 0:
        unstructured_square_mesh_gen = UM2N.UnstructuredSquareMeshGenerator(
            scale=scale_x, mesh_type=mesh_type
        )  # noqa
        return unstructured_square_mesh_gen.generate_mesh(
            res=lc,
            output_filename=os.path.join(problem_mesh_dir, f"mesh_{i:04d}.msh"),
        )
    else:
        n_grid = int(1 / lc)
        return fd.UnitSquareMesh(n_grid, n_grid)


def process_features(parameters, problem_data_dir):

    # create mesh
    mesh = create_mesh(
        i, mesh_type = parameters["mesh_type"], lc = parameters["lc"],
        scale_x = parameters["scale_x"], problem_mesh_dir = directories["mesh"]
    )
    # Generate Random solution field
    rand_u_generator = UM2N.RandSourceGenerator(
        use_iso= parameters["data_type"] == "iso",
        dist_params = parameters
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

    grad_uh_interpolate = fd.assemble(interpolate(fd.grad(uh),
                                fd.VectorFunctionSpace(mesh, "CG", 1)
                                ))
    grad_norm = fd.Function(res["function_space"])
    grad_norm.project(grad_uh_interpolate[0] ** 2 + grad_uh_interpolate[1] ** 2)
    grad_norm /= grad_norm.vector().max()

    # FOR OUTPUT
    # RHS of helmholtz problem
    f_rhs = fd.assemble(interpolate(helmholtz_eq.f, helmholtz_eq.function_space))


    # generate mesh?

    mesh_gen = UM2N.MeshGenerator(params={"eq": helmholtz_eq, "mesh": mesh})
    monitor_val = mesh_gen.monitor_func(mesh)
    hessian = mesh_gen.get_hessian(mesh)
    hessian_norm = fd.project(mesh_gen.get_hessian_norm(mesh),
                                fd.FunctionSpace(mesh, "CG", 1)
                                )

    # move the mesh?
    start = time.perf_counter()
    new_mesh = mesh_gen.move_mesh()  # noqa
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
        use_4_edge=True,
        feature={
            "uh": uh.dat.data_ro.reshape(-1, 1),
            "grad_uh": grad_uh_interpolate.dat.data_ro.reshape(-1, 2),
            "grad_uh_norm": grad_norm.dat.data_ro.reshape(-1, 1),
            "hessian": hessian.dat.data_ro.reshape(-1, 4),
            "hessian_norm": hessian_norm.dat.data_ro.reshape(-1, 1),
            "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
            "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
            "phi": phi.dat.data_ro.reshape(-1, 1),
            "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
            "f": f_rhs.dat.data_ro.reshape(-1, 1),
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

    # save out data
    mesh_processor.save_taining_data(
        os.path.join(problem_data_dir, f"data_{i:04d}")
    )

    high_res_mesh = create_mesh(
        i, mesh_type = parameters["mesh_type"], lc = 1e-2,
        scale_x = parameters["scale_x"], problem_mesh_dir = directories["mesh_fine"]
    )

    res_high_res = helmholtz_eq.discretise(high_res_mesh)
    u_exact = fd.assemble(interpolate(res_high_res["u_exact"],
                            res_high_res["function_space"])
                            )

    uh_proj = fd.project(uh, fd.FunctionSpace(high_res_mesh, "CG", 1))
    uh_new_proj = fd.project(uh_new, fd.FunctionSpace(high_res_mesh, "CG", 1))

    error_original_mesh = fd.errornorm(u_exact, uh_proj)
    error_optimal_mesh = fd.errornorm(u_exact, uh_new_proj)

    # Write to CSV
    with open(os.path.join(directories["log"], f"log_{i:04d}.csv"), mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header (keys)
        csv_writer.writerow(["error_og", "error_adapt", "time"])
        # Write data (values)
        csv_writer.writerow([error_original_mesh, error_optimal_mesh, dur])

    print("error og/optimal:", error_original_mesh, error_optimal_mesh)

    # ====  Plot mesh, solution, error ======================

    rows, cols = 3, 3
    cmap = "seismic"

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

    # Error on high resolution mesh
    cb = fd.tripcolor(monitor_val, cmap=cmap, axes=ax[2, 0])
    ax[2, 0].set_title("Monitor values")
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

    fig.savefig(os.path.join(directories["plot_compare"], f"plot_{i:04d}.png"))
    plt.close()

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

    print(f"Parameters saved to {csv_file_path}")

if __name__ == "__main__":

    # parse args
    args = parse_arguments()
    
    # ====  Parameters ======================
    parameters = {
        # parameters for problem
        "problem": "helmholtz",
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
        "c_min": 0.2 if args.boundary_scheme == "pad" else 0,
        "c_max": 0.8 if args.boundary_scheme == "pad" else 1,
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

    # Initialize directories
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # QC:
    print(f"Project Directory: {project_dir}")

    dataset_dir = os.path.join(
        project_dir, "data", f"dataset_meshtype_{args.mesh_type}", "helmholtz"
    )
    problem_specific_dir = os.path.join(
        dataset_dir, "z=<{},{}>_ndist={}_max_dist={}_lc={}_n={}_{}_{}_meshtype_{}".format(
            parameters["z_min"], parameters["z_max"],
            parameters["n_dist"],parameters["max_dist"],
            parameters["lc"], parameters["n_samples"],
            parameters["data_type"], parameters["scheme"], parameters["mesh_type"]
        ),
    )
    subdirs = [
        "data", "plot", "plot_compare", "log", "mesh", "mesh_fine",
        "train", "test", "val",
    ]

    # setup directory structure
    directories = setup_and_clear_directories(problem_specific_dir,subdirs)

    # output parameters to csv
    output_csv(parameters, [
        "cmin","cmax", "sigma_mean_scaler", "sigma_sigma_scaler", "sigma_eps"
        "data_type", "scheme", "n_samples", "lc", "mesh_type"
        ],
        problem_specific_dir
        )

    # Generate samples
    for i in range(parameters["n_samples"]):
        try:
            print(f"Generating Sample: {i}")

            # create dataset
            process_features(parameters, directories["data"])

        except fd.exceptions.ConvergenceError:
            print(f"Iteration {i} did not converge.")
            continue

    # Split data into train, test, and validation sets
    num_train = int(parameters["n_samples"] * parameters["p_train"])
    num_test = int(parameters["n_samples"] * parameters["p_test"])
    num_val = parameters["n_samples"] - num_train - num_test

    move_data(directories["train"], directories["data"], 0, num_train)
    move_data(directories["test"], directories["data"], num_train, num_train + num_test)
    move_data(directories["val"], directories["data"], num_train + num_test, num_train + num_test + num_val)
