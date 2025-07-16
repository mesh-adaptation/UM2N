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
        help="Use padded scheme or full-scale scheme to sample central point of the bump (pad/full).",
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

    return parser.parse_args()


def process_features(parameters, problem_data_dir):
    # create mesh
    scale_x = parameters["scale_x"]
    mesh_type = parameters["mesh_type"]
    lc = parameters["lc"]
    unstructured_square_mesh_gen = UM2N.UnstructuredSquareMeshGenerator(
        scale=scale_x, mesh_type=mesh_type
    )  # noqa
    mesh = unstructured_square_mesh_gen.generate_mesh(
        res=lc, output_filename=os.path.join(directories["mesh"], f"mesh{i}.msh")
    )
    # Generate Random solution field
    rand_u_generator = UM2N.RandSourceGenerator(
        use_iso=parameters["data_type"] == "iso", dist_params=parameters
    )

    # generate equation
    poisson_eq = UM2N.RandPoissonEqGenerator(rand_u_generator)
    # discretise the equation
    res = poisson_eq.discretise(mesh)
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

    mesh_gen = UM2N.MeshGenerator(params={"eq": poisson_eq, "mesh": mesh})
    monitor_val = mesh_gen.monitor_func(mesh)
    hessian = mesh_gen.get_hessian(mesh)
    hessian_norm = fd.project(
        mesh_gen.get_hessian_norm(mesh), fd.FunctionSpace(mesh, "CG", 1)
    )

    func_vec_space = fd.VectorFunctionSpace(mesh, "CG", 1)
    grad_uh_interpolate = fd.assemble(interpolate(fd.grad(uh), func_vec_space))

    grad_norm = fd.Function(res["function_space"])
    grad_norm.project(grad_uh_interpolate[0] ** 2 + grad_uh_interpolate[1] ** 2)
    grad_norm /= grad_norm.vector().max()

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
    new_res = poisson_eq.discretise(new_mesh)
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

    mesh_processor.save_taining_data(os.path.join(directories["data"], f"data_{i:04d}"))

    # ====  Plot Scripts ======================
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    # Plot the exact solution
    ax1.set_title("Exact Solution")
    fd.trisurf(
        fd.assemble(interpolate(res["u_exact"], res["function_space"])), axes=ax1
    )
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

    fig.savefig(os.path.join(directories["plot"], "plot_{}.png".format(i)))
    # ==========================================

    # generate log file
    high_res_mesh = unstructured_square_mesh_gen.generate_mesh(
        res=1e-2,
        output_filename=os.path.join(directories["mesh"], f"mesh{i}.msh"),
    )
    high_res_function_space = fd.FunctionSpace(high_res_mesh, "CG", 1)

    res_high_res = poisson_eq.discretise(high_res_mesh)
    u_exact = fd.assemble(
        interpolate(res_high_res["u_exact"], res_high_res["function_space"])
    )

    uh = fd.project(uh, high_res_function_space)
    uh_new = fd.project(uh_new, high_res_function_space)

    error_original_mesh = fd.errornorm(u_exact, uh)
    error_optimal_mesh = fd.errornorm(u_exact, uh_new)

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


if __name__ == "__main__":
    # parse args
    args = parse_arguments()

    # ====  Parameters ======================
    parameters = {
        # parameters for problem
        "problem": "poisson",
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
    key_list = ["cmin", "cmax", "data_type", "scheme", "n_samples", "lc", "mesh_type"]
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
