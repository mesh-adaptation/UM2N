# Author: Chunyang Wang
# GitHub Username: chunyang-w

from argparse import ArgumentParser

import firedrake as fd
import matplotlib.pyplot as plt
from build_helper import *

import UM2N


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Build Burgers dataset with square meshes.")
    parser.add_argument(
        "--mesh_type", type=int, default=2, help="Algorithm used to generate mesh."
    )
    parser.add_argument(
        "--max_dist", type=int, default=6, help="Max number of distributions."
    )
    parser.add_argument(
        "--n_dist", type=int, default=None, help="Number of distributions."
    )
    parser.add_argument(
        "--lc", type=float, default=6e-2, help="Length characteristic of mesh elements."
    )
    parser.add_argument(
        "--field_type", type=str, default="iso", help="Data type (aniso/iso)."
    )
    parser.add_argument(
        "--boundary_scheme",
        type=str,
        default="pad",
        help="use padded scheme or full-scale scheme to sample central point of the bump (pad/full).",
    )
    parser.add_argument(
        "--n_case", type=int, default=5, help="Number of simulation cases."
    )
    parser.add_argument(
        "--n_grid",
        type=int,
        default=20,
        help="Number of grids for uniform mesh if mesh_type 0.",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=63,
        help="number of samples generated / Random seed for reproducibility.",
    )

    parsed_args = parser.parse_args()

    # Handle dependency between max_dist and n_dist
    # max number of distributions used to generate the dataset
    # only if n_dist is not set if n_dist is set, max_dist will be disabled
    if parsed_args.n_dist is not None:
        parsed_args.max_dist = None  # Disable max_dist if n_dist is set
        print("Warning: max_dist is ignored because n_dist is set.")
    # QC:
    # print(parsed_args)

    return parser.parse_args()


def generate_mesh(parameters, directories):
    """Generate the mesh based on the specified type."""
    if parameters["mesh_type"] != 0:
        mesh_gen = UM2N.UnstructuredSquareMeshGenerator(
            scale=parameters["scale_x"], mesh_type=parameters["mesh_type"]
        )
        mesh = mesh_gen.generate_mesh(
            res=parameters["lc"],
            output_filename=os.path.join(directories["mesh"], "mesh.msh"),
        )
        mesh_new = fd.Mesh(os.path.join(directories["mesh"], "mesh.msh"))
        mesh_fine = mesh_gen.generate_mesh(
            res=1e-2, output_filename=os.path.join(directories["mesh_fine"], "mesh.msh")
        )
    else:
        n_grid = parameters["n_grid"]
        mesh = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_new = fd.UnitSquareMesh(n_grid, n_grid)
        mesh_fine = fd.UnitSquareMesh(100, 100)
    return mesh, mesh_new, mesh_fine


def get_sample_param_of_nu_generalization_by_idx_train(idx_in):
    """
    Retrieve sample parameters for the Burgers problem based on the given index.

    Args:
        idx_in (int): Index of the sample.

    Returns:
        tuple: A list of Gaussian parameters and the viscosity value (nu).
    """
    # Define a mapping of indices to parameters
    param_map = {
        1: ({"cx": 0.225, "cy": 0.5, "w": 0.01}, 0.0001),
        2: ({"cx": 0.225, "cy": 0.5, "w": 0.01}, 0.001),
        3: ({"cx": 0.225, "cy": 0.5, "w": 0.01}, 0.002),
        4: (
            [{"cx": 0.3, "cy": 0.35, "w": 0.01}, {"cx": 0.15, "cy": 0.65, "w": 0.01}],
            0.0001,
        ),
        5: (
            [{"cx": 0.3, "cy": 0.35, "w": 0.01}, {"cx": 0.15, "cy": 0.65, "w": 0.01}],
            0.001,
        ),
        6: (
            [{"cx": 0.3, "cy": 0.35, "w": 0.01}, {"cx": 0.15, "cy": 0.65, "w": 0.01}],
            0.002,
        ),
        7: (
            [
                {"cx": 0.3, "cy": 0.7, "w": 0.01},
                {"cx": 0.3, "cy": 0.3, "w": 0.01},
                {"cx": 0.15, "cy": 0.5, "w": 0.01},
            ],
            0.0001,
        ),
        8: (
            [
                {"cx": 0.3, "cy": 0.7, "w": 0.01},
                {"cx": 0.3, "cy": 0.3, "w": 0.01},
                {"cx": 0.15, "cy": 0.5, "w": 0.01},
            ],
            0.001,
        ),
        9: (
            [
                {"cx": 0.3, "cy": 0.7, "w": 0.01},
                {"cx": 0.3, "cy": 0.3, "w": 0.01},
                {"cx": 0.15, "cy": 0.5, "w": 0.01},
            ],
            0.002,
        ),
    }

    # Retrieve the parameters and viscosity for the given index
    if idx_in not in param_map:
        raise ValueError(
            f"Invalid index: {idx_in}. Supported indices are {list(param_map.keys())}."
        )

    params, nu_ = param_map[idx_in]
    # Ensure params is always a list
    gauss_list_ = params if isinstance(params, list) else [params]

    return gauss_list_, nu_


def sample_from_loop(
    uh,
    uh_grad,
    grad_norm,
    hessian,
    hessian_norm,
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
    nu,
    gauss_list,
    t,
    idx,
    monitor_val,
    error_og_list=[],
    error_adapt_list=[],
):
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
            "grad_uh_norm": grad_norm.dat.data_ro.reshape(-1, 1),
            "hessian": hessian.dat.data_ro.reshape(-1, 4),
            "hessian_norm": hessian_norm.dat.data_ro.reshape(-1, 1),
            "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
            "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
            "phi": phi.dat.data_ro.reshape(-1, 1),
            "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
            "monitor_val": monitor_val.dat.data_ro.reshape(-1, 1),
        },
        raw_feature={
            "uh": uh,
            "hessian_norm": hessian_norm,
            "monitor_val": monitor_val,
            "jacobian": jacobian,
            "jacobian_det": jacobian_det,
        },
        nu=nu,
        gauss_list=gauss_list,
        dur=dur,
        t=t,
        idx=idx,
    )

    mesh_processor.save_taining_data(os.path.join(directories["data"], f"data_{i:04d}"))

    # ====  Plot Scripts ======================
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    # Plot the exact solution
    ax1.set_title("Solution field (HR)")
    fd.trisurf(uh_fine, axes=ax1)
    # Plot the solved solution
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    ax2.set_title("Solution field (Original Mesh)")
    fd.trisurf(uh, axes=ax2)

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.set_title("Solution field (Adapted Mesh)")
    fd.trisurf(uh_new, axes=ax3)

    # Plot the mesh
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("Original Mesh")
    fd.triplot(mesh, axes=ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("Adapted Mesh")
    fd.triplot(mesh_new, axes=ax5)

    # plot mesh with function evaluated on it
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("Soultion Projected on optimal mesh")
    fd.tripcolor(uh_new, cmap="coolwarm", axes=ax6)
    fd.triplot(mesh_new, axes=ax6)

    fig.savefig(os.path.join(directories["plot"], "plot_{}.png".format(i)))
    i += 1

    # ==========================================
    uh = fd.project(uh, function_space_fine)
    uh_new = fd.project(uh_new, function_space_fine)

    error_original_mesh = fd.errornorm(uh, uh_fine, norm_type="L2")
    error_optimal_mesh = fd.errornorm(uh_new, uh_fine, norm_type="L2")

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
    return


if __name__ == "__main__":
    # parse args
    args = parse_arguments()

    # ====  Parameters ======================
    parameters = {
        # parameters for problem
        "problem": "burgers",
        "n_case": args.n_case,
        # parameters for random source
        "n_dist": args.n_dist,
        "max_dist": args.max_dist,
        "lc": args.lc,
        "n_grid": args.n_grid,
        # parameters for mesh def
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
        # "sigma_mean_scaler": 1 / 4,
        # "sigma_sigma_scaler": 1 / 6,
        # "sigma_eps": 1 / 8,
        # parameters for data split
        "p_train": 0.75,
        "p_test": 0.15,
        "p_val": 0.1,
    }

    # Set random seed
    random.seed(args.rand_seed)

    # ====  Setup Directories ======================
    problem_specific_dir = "lc={lc}_ngrid_{n_grid}_n={n_case}_{data_type}_{scheme}_meshtype_{mesh_type}".format(
        lc=parameters["lc"],
        n_grid=parameters["n_grid"],
        n_case=parameters["n_case"],
        data_type=parameters["data_type"],
        scheme=parameters["scheme"],
        mesh_type=parameters["mesh_type"],
    )

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

    directories = setup_directories(
        problem=parameters["problem"],
        mesh_type=parameters["mesh_type"],
        base_dir=None,
        subdirs=subdirs,
        dir_format=problem_specific_dir,
    )

    # ====  Output CSV ======================
    key_list = ["cmin", "cmax", "data_type", "scheme", "lc", "mesh_type"]
    output_csv(parameters, key_list, directories["log"])

    # ====  Data Generation Scripts ======================

    i = 0

    # QC:
    print("In build_dataset.py")
    # for idx in range(1, n_case + 1):
    for idx in range(1, parameters["n_case"] + 1):
        try:
            # QC:
            print(f"Case {idx} building ...")
            mesh, mesh_new, mesh_fine = generate_mesh(parameters, directories)
            # Generate Random solution field
            gaussian_list, nu = get_sample_param_of_nu_generalization_by_idx_train(idx)  # noqa
            solver = UM2N.BurgersSolver(
                mesh, mesh_fine, mesh_new, gauss_list=gaussian_list, nu=nu, idx=idx
            )
            solver.solve_problem(sample_from_loop)
            print()
        except fd.exceptions.ConvergenceError:
            print("ConvergenceError")
            pass
    print("Done!")


# ====  Data Generation Scripts ======================
