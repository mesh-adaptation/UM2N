# Author: Chunyang Wang
# GitHub Username: chunyang-w

import time
from argparse import ArgumentParser

import firedrake as fd
import matplotlib.pyplot as plt
from build_helper import *
from firedrake.__future__ import interpolate

import UM2N


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--mesh_type", type=int, default=2, help="Algorithm used to generate mesh"
    )
    parser.add_argument(
        "--max_dist", type=int, default=6, help="Max number of distributions"
    )
    parser.add_argument(
        "--n_dist", type=int, default=None, help="Number of distributions"
    )
    parser.add_argument(
        "--lc", type=float, default=5e-2, help="Length characteristic of mesh elements"
    )
    parser.add_argument(
        "--field_type", type=str, default="aniso", help="Data type (aniso/iso)"
    )
    parser.add_argument(
        "--boundary_scheme",
        type=str,
        default="full",
        help="Use padded scheme or full-scale scheme to sample central point of the bump (pad/full)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples generated"
    )
    parser.add_argument("--rand_seed", type=int, default=63, help="Random seed")

    parsed_args = parser.parse_args()

    # Handle dependency between max_dist and n_dist
    # max number of distributions used to generate the dataset
    # only if n_dist is not set if n_dist is set, max_dist will be disabled
    if parsed_args.n_dist is not None:
        parsed_args.max_dist = None  # Disable max_dist if n_dist is set
        print("Warning: max_dist is ignored because n_dist is set.")
    # QC:
    # print(parsed_args)

    return parsed_args


def setup_directories(problem, mesh_type, base_dir=None, subdirs=None, dir_format=None):
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
    dataset_dir = os.path.join(
        project_dir, "data", f"dataset_meshtype_{mesh_type}", problem
    )

    # Use the provided format string for the problem-specific directory
    if dir_format is None:
        problem_specific_dir = os.path.join(
            dataset_dir, f"{problem}_meshtype_{mesh_type}"
        )
    else:
        # check if dir_format is a valid string format
        if not isinstance(dir_format, str):
            raise ValueError("dir_format must be a string.")
        problem_specific_dir = os.path.join(dataset_dir, dir_format)

    # Define default subdirectories if not provided
    if subdirs is None:
        subdirs = [
            "data",
            "plot",
            "log",
            "mesh",
            "mesh_fine",
            "plot_compare",
            "train",
            "test",
            "val",
        ]

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
    # print(f"Subdirectories created: {directories}")

    return directories


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


def process_features(parameters, directories):
    # create mesh
    mesh = create_mesh(
        i,
        mesh_type=parameters["mesh_type"],
        lc=parameters["lc"],
        scale_x=parameters["scale_x"],
        problem_mesh_dir=directories["mesh"],
    )
    # Generate Random solution field
    rand_u_generator = UM2N.RandSourceGenerator(
        use_iso=parameters["data_type"] == "iso", dist_params=parameters
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

    grad_uh_interpolate = fd.assemble(
        interpolate(fd.grad(uh), fd.VectorFunctionSpace(mesh, "CG", 1))
    )
    grad_norm = fd.Function(res["function_space"])
    grad_norm.project(grad_uh_interpolate[0] ** 2 + grad_uh_interpolate[1] ** 2)
    grad_norm /= grad_norm.vector().max()

    # FOR OUTPUT
    # RHS of helmholtz problem
    f_rhs = fd.assemble(interpolate(helmholtz_eq.f, helmholtz_eq.function_space))

    # generate mesh
    mesh_gen = UM2N.MeshGenerator(params={"eq": helmholtz_eq, "mesh": mesh})
    monitor_val = mesh_gen.monitor_func(mesh)
    hessian = mesh_gen.get_hessian(mesh)
    hessian_norm = fd.project(
        mesh_gen.get_hessian_norm(mesh), fd.FunctionSpace(mesh, "CG", 1)
    )

    # move the mesh
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
    mesh_processor.save_taining_data(os.path.join(directories["data"], f"data_{i:04d}"))

    # ====  Log File ============================================
    high_res_mesh = create_mesh(
        i,
        mesh_type=parameters["mesh_type"],
        lc=1e-2,
        scale_x=parameters["scale_x"],
        problem_mesh_dir=directories["mesh_fine"],
    )

    res_high_res = helmholtz_eq.discretise(high_res_mesh)
    u_exact = fd.assemble(
        interpolate(res_high_res["u_exact"], res_high_res["function_space"])
    )

    uh_proj = fd.project(uh, fd.FunctionSpace(high_res_mesh, "CG", 1))
    uh_new_proj = fd.project(uh_new, fd.FunctionSpace(high_res_mesh, "CG", 1))

    error_original_mesh = fd.errornorm(u_exact, uh_proj)
    error_optimal_mesh = fd.errornorm(u_exact, uh_new_proj)

    # Write to CSV
    with open(
        os.path.join(directories["log"], f"log_{i:04d}.csv"), mode="w", newline=""
    ) as csvfile:
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
        # parameters for random source
        "n_dist": args.n_dist,
        "max_dist": args.max_dist,
        "lc": args.lc,
        # parameters for mesh def
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

    # ====  Setup Directories ======================
    problem_specific_dir = (
        "z=<{},{}>_ndist={}_max_dist={}_lc={}_n={}_{}_{}_meshtype_{}".format(
            parameters["z_min"],
            parameters["z_max"],
            parameters["n_dist"],
            parameters["max_dist"],
            parameters["lc"],
            parameters["n_samples"],
            parameters["data_type"],
            parameters["scheme"],
            parameters["mesh_type"],
        )
    )

    subdirs = [
        "data",
        "plot",
        "plot_compare",
        "log",
        "mesh",
        "mesh_fine",
        "train",
        "test",
        "val",
    ]

    directories = setup_directories(
        problem=parameters["problem"],
        mesh_type=parameters["mesh_type"],
        base_dir=None,
        subdirs=subdirs,
        dir_format=problem_specific_dir,
    )

    # ====  Output CSV ======================
    key_list = [
        "cmin",
        "cmax",
        "sigma_mean_scaler",
        "sigma_sigma_scaler",
        "sigma_eps" "data_type",
        "scheme",
        "n_samples",
        "lc",
        "mesh_type",
    ]
    output_csv(parameters, key_list, directories["log"])

    # ====  Data Generation Scripts ======================
    for i in range(parameters["n_samples"]):
        try:
            print(f"Generating Sample: {i}")

            # create dataset
            process_features(parameters, directories)

        except fd.exceptions.ConvergenceError:
            print(f"Iteration {i} did not converge.")
            continue

    # ====  Data Splits ============================================
    # TODO: this should probably be done in the training script, not the build script
    split_data(
        source_dir=directories["data"],
        train_dir=directories["train"],
        test_dir=directories["test"],
        val_dir=directories["val"],
        train_ratio=parameters["p_train"],
        test_ratio=parameters["p_test"],
        val_ratio=parameters["p_val"],
    )
