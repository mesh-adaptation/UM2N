# Author: Chunyang Wang
# GitHub Username: chunyang-w
import csv
import os
import shutil
from argparse import ArgumentParser

import firedrake as fd
import matplotlib.pyplot as plt
import pandas as pd

# import UM2N

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
    parser.add_argument("--sigma", type=float, default=(0.05 / 3), help="initial ring shape control")
    parser.add_argument("--r_0", type=float, default=0.2, help="initial ring radius")
    parser.add_argument("--x_0", type=float, default=0.5, help="ring center x coordinate")
    parser.add_argument("--y_0", type=float, default=0.5, help="ring center y coordinate")
    parser.add_argument("--alpha", type=float, default=1.5, help="swirl (velocity) scalar coefficient")
    parser.add_argument("--save_interval", type=int, default=10, help="output sample file interval")
    parser.add_argument("--lc", type=float, default=5e-2, help="Length characteristic of unstructured mesh elements.")
    parser.add_argument("--n_grid", type=int, default=20, help="number number of grids in a mesh when mesh_type is 0)")
    parser.add_argument("--n_monitor_smooth", type=int, default=10, help="apply Laplacian smoother n time to monitor function")
   
    
    parsed_args = parser.parse_args()

    # Handle dependency between max_dist and n_dist
    # max number of distributions used to generate the dataset
    # only if n_dist is not set if n_dist is set, max_dist will be disabled
    # if parsed_args.n_dist is not None:
    #     parsed_args.max_dist = None  # Disable max_dist if n_dist is set
    #     print("Warning: max_dist is ignored because n_dist is set.")
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
    x_0, # ej321 - added x_0
    y_0, # ej321 - added y_0
    t,
    error_og_list=[],
    error_adapt_list=[],
):
    """
    Call back function for storing data.
    """
    global i
    print("before processing")
    mesh_processor = UM2N.MeshProcessor(
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
            "uh_adapt": uh_new.dat.data_ro.reshape(-1, 1),
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

    mesh_processor.save_taining_data(os.path.join(directories["data"], f"data_{i:04d}"))

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

    # ====  Log File ============================================
    # function_space_fine = fd.FunctionSpace(mesh_fine, 'CG', 1)
    uh_proj = fd.project(uh, function_space_fine)
    uh_new_proj = fd.project(uh_new, function_space_fine)

    error_original_mesh = fd.errornorm(uh_proj, uh_fine, norm_type="L2")
    error_optimal_mesh = fd.errornorm(uh_new_proj, uh_fine, norm_type="L2")

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
    fig, ax = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), layout="compressed"
    )

    # High resolution mesh
    fd.triplot(mesh_fine, axes=ax[0, 0])
    ax[0, 0].set_title("High resolution Mesh (100 x 100)")
    # Orginal low resolution uniform mesh
    fd.triplot(mesh_og, axes=ax[0, 1])
    ax[0, 1].set_title("Original uniform Mesh")
    # Adapted mesh
    fd.triplot(mesh_new, axes=ax[0, 2])
    ax[0, 2].set_title("Adapted Mesh (MA)")

    cmap = "seismic"
    # Solution on high resolution mesh
    cb = fd.tripcolor(uh_fine, cmap=cmap, axes=ax[1, 0])
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
    ax[2, 0].set_title("Monitor Values")
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

    fig.savefig(os.path.join(directories["plot_compare"], f"plot_{i:04d}.png"))
    plt.close()
    i += 1
    return


if __name__ == "__main__":

    # parse args
    args = parse_arguments()
    
    # ====  Parameters ======================
    parameters = {
        # parameters for problem
        "problem": "swirl",
        # parameters for simulation time & time steps
        "T": 1,
        "dt": 1e-3,  # The CFL condition requires that the timestep is less than 0.0014 for fine mesh
        "n_step": 1000,
        # "n_case": args.n_case, # burgers problem only
        # parameters for random source
        # "n_dist": args.n_dist,
        # "max_dist": args.max_dist,
        "lc": args.lc,
        "n_grid": args.n_grid if args.n_grid else int(1 / lc),
        # parameters for ??????
        # "n_samples": args.n_samples,
        # "data_type": args.field_type,
        # "scheme": args.boundary_scheme,
        "mesh_type": int(args.mesh_type),
        "n_monitor_smooth": args.n_monitor_smooth,
        # parameters for domain scale
        "scale_x": 1,
        "scale_y": 1,
        # parameters for anisotropic data - distribution height scaler
        # "z_max": 1,
        # "z_min": 0,
        # parameters for ?????
        # "x_start": 0,
        # "x_end": 1,
        # "y_start": 0,
        # "y_end": 1,
        # parameters for initial condition
        "sigma": args.sigma,
        "r_0": args.r_0,
        "x_0": args.x_0,
        "y_0": args.y_0,
        "alpha": args.alpha,
        # parameters for storing files
        "save_interval": args.save_interval,
        "fail_t": [],  # list storing failing dts
        
        # parameters for isotropic data
        # "w_min": 0.05,
        # "w_max": 0.2,
        # "c_min": 0.2 if args.boundary_scheme == "pad" else 0,
        # "c_max": 0.8 if args.boundary_scheme == "pad" else 1,
        # parameters for dataset challenging level
        # larger, less challenging (because the gaussian is more like a circle)
        # "sigma_mean_scaler": 1 / 4,
        # "sigma_sigma_scaler": 1 / 6,
        # "sigma_eps": 1 / 8,
        # parameters for data split
        # "p_train": 0.75,
        # "p_test": 0.15,
        # "p_val": 0.1,
    }

    # # Set random seed
    # random.seed(args.rand_seed)

    # ====  Setup Directories ======================
    problem_specific_dir = "sigma_{:.3f}_alpha_{}_r0_{}_x0_{}_y0_{}_lc_{}_ngrid_{}_interval_{}_meshtype_{}_smooth_{}".format(
            parameters["sigma"], parameters["alpha"],
            parameters["r_0"], parameters["x_0"], parameters["y_0"],
            parameters["lc"], parameters["n_grid"],
            parameters["save_interval"], parameters["mesh_type"],
            parameters["n_monitor_smooth"]
    )

    subdirs = [
        "data", "plot","plot_compare","log", "mesh", "mesh_fine",
        # "train", "test", "val",
    ]

    directories = setup_directories(problem = parameters["problem"],
                        mesh_type = parameters["mesh_type"],
                        base_dir = None,
                        subdirs = subdirs,
                        dir_format = problem_specific_dir)


    # ====  Output CSV ======================
    key_list = [
            "sigma",
            "alpha",
            "r_0",
            "x_0",
            "y_0",
            "save_interval",
            "T",
            "n_step",
            "dt",
            "fail_t",
            "lc",
            "fail_cases",
            "mesh_type",
    ]
    output_csv(parameters, key_list, directories["data"])

    # ====  Data Generation Scripts ======================
    print("In build_dataset.py")

    i = 0  # global variable to count the number of samples
    mesh = None
    mesh_fine = None
    mesh_new = None
    mesh_type = parameters["mesh_type"]
    lc = parameters["lc"]
    n_grid = parameters["n_grid"]
    if mesh_type != 0:
        mesh_gen = UM2N.UnstructuredSquareMeshGenerator(mesh_type=mesh_type)
        mesh = mesh_gen.generate_mesh(
            res=lc, output_filename=os.path.join(directories["mesh"], "mesh.msh")
        )
        mesh_new = mesh_gen.generate_mesh(
            res=lc, output_filename=os.path.join(directories["mesh"], "mesh.msh")
        )
        mesh_model = mesh_gen.generate_mesh(
            res=lc, output_filename=os.path.join(directories["mesh"], "mesh.msh")
        )
        # ej321 - is this extra call to mesh gen needed?
        mesh_gen_fine = UM2N.UnstructuredSquareMeshGenerator(mesh_type=mesh_type)
        mesh_fine = mesh_gen_fine.generate_mesh(
            res=1e-2, output_filename=os.path.join(directories["mesh_fine"], "mesh.msh")
        )
    else:
        mesh = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_new = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_model = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_fine = fd.UnitSquareMesh(100, 100)

    # solver defination
    swirl_solver = UM2N.SwirlSolver(
        mesh,
        mesh_fine,
        mesh_new,
        mesh_model=mesh_model,
        **parameters
        # sigma=sigma,
        # alpha=alpha,
        # r_0=r_0,
        # x_0=x_0,
        # y_0=y_0,
        # save_interval=save_interval,
        # T=T,
        # dt=dt,
        # n_step=n_step,
        # n_monitor_smooth=n_monitor_smooth,
    )

    swirl_solver.solve_problem(callback=sample_from_loop, fail_callback=fail_callback)

    print("Done!")

