import time
from argparse import ArgumentParser

import firedrake as fd
import matplotlib.pyplot as plt
import movement as mv
from build_helper import *
from matplotlib.colors import LogNorm

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


def generate_monitor():
    """
    Generate a monitor function and its parameters.

    Returns:
        monitor_func: A Function(mesh) which returns a Firedrake Form of the monitor function eq
        monitor_params: A dictionary containing the parameters used to generate the monitor.
    """
    # Generate random monitor parameters
    # Note: These parameters are specific to the RingMonitor function
    monitor_params = {
        "centre": (
            round(random.uniform(0.2, 0.8), 3),  # Random x-coordinate of the center
            round(random.uniform(0.2, 0.8), 3),  # Random y-coordinate of the center
        ),
        "radius": round(random.uniform(0.1, 0.5), 3),  # Random radius
        "amplitude": int(random.uniform(10, 100)),  # Random amplitude
        "width": int(random.uniform(10, 200)),  # Random width
    }

    # Initialize the monitor function
    # The monitor function is created using the RingMonitorBuilder from the movement library.
    # To modify this function for a different monitor, replace.
    mb = mv.RingMonitorBuilder(
        centre=monitor_params["centre"],
        radius=monitor_params["radius"],
        amplitude=monitor_params["amplitude"],
        width=monitor_params["width"],
    )
    # Get the monitor function as a Firedrake Form
    monitor_func = mb.get_monitor()

    return monitor_func, monitor_params


def process_features(parameters, directories):
    # ====  Create the mesh ======================
    mesh = create_mesh(
        i,
        mesh_type=parameters["mesh_type"],
        lc=parameters["lc"],
        scale_x=parameters["scale_x"],
        problem_mesh_dir=directories["mesh"],
    )

    # ====  Generate monitor ======================
    monitor_func, monitor_params = generate_monitor()

    # output specific parameters to csv file
    output_csv(monitor_params, list(monitor_params.keys()), directories["log"])

    # get projection of the monitor function for feature output
    monitor_val = monitor_func(mesh)

    # ====  Move the mesh ======================

    # create Monge Ampere obj
    mover = mv.MongeAmpereMover(
        mesh, monitor_func, method="relaxation", rtol=1e-3, maxiter=500
    )

    start = time.perf_counter()

    # move the mesh
    mover.move()

    # assign new_mesh
    new_mesh = mover.mesh

    # ====  Extract features from moved mesh ======================

    # this is the jacobian of x with respect to xi
    jacobian = fd.project(
        fd.Identity(2) + mover.H, fd.TensorFunctionSpace(new_mesh, "CG", 1)
    )
    jacobian_det = fd.Function(fd.FunctionSpace(new_mesh, "CG", 1), name="jacobian_det")
    jacobian_det.project(
        jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
    )

    # get phi/grad_phi projected to the original mesh
    phi = mover.phi
    grad_phi = mover.grad_phi

    end = time.perf_counter()
    dur = (end - start) * 1000

    # ====  Process data for training ======================
    mesh_processor = UM2N.MeshProcessor(
        original_mesh=mesh,
        optimal_mesh=new_mesh,
        function_space=fd.FunctionSpace(new_mesh, "CG", 1),
        use_4_edge=True,
        feature={
            "jacobian": jacobian.dat.data_ro.reshape(-1, 4),
            "jacobian_det": jacobian_det.dat.data_ro.reshape(-1, 1),
            "phi": phi.dat.data_ro.reshape(-1, 1),
            "grad_phi": grad_phi.dat.data_ro.reshape(-1, 2),
            "monitor_val": monitor_val.dat.data_ro.reshape(-1, 1),
        },
        raw_feature={
            "monitor_val": monitor_val,
            "jacobian": jacobian,
            "jacobian_det": jacobian_det,
        },
        # dist_params=None, # When nothing passed the default is used
    )

    # save out data
    mesh_processor.save_taining_data(os.path.join(directories["data"], f"data_{i:04d}"))

    # ====  Plot mesh, solution, error ======================
    rows, cols = 2, 2
    cmap = "plasma"

    fig, ax = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), layout="compressed"
    )

    # Orginal low resolution uniform mesh
    fd.triplot(mesh, axes=ax[0, 0])
    ax[0, 0].set_title("Original uniform Mesh")
    # Adapted mesh
    fd.triplot(new_mesh, axes=ax[0, 1])
    ax[0, 1].set_title(f"Adapted Mesh (MA): time taken {dur:.2f} ms")

    # Monitor on high resolution mesh
    fd.triplot(mesh, axes=ax[1, 0])
    cb = fd.tripcolor(monitor_val, cmap=cmap, axes=ax[1, 0], alpha=0.5, norm=LogNorm())
    ax[1, 0].set_title(
        f'Monitor (c: {monitor_params["centre"]} r: {monitor_params["radius"]} a: {monitor_params["amplitude"]} w: {monitor_params["width"]} )'
    )
    plt.colorbar(cb)

    # Monitor on high resolution mesh
    fd.triplot(new_mesh, axes=ax[1, 1])
    cb = fd.tripcolor(monitor_val, cmap=cmap, axes=ax[1, 1], alpha=0.5, norm=LogNorm())
    ax[1, 1].set_title("Monitor overlayed with Adapted Mesh")
    plt.colorbar(cb)

    for rr in range(rows):
        for cc in range(cols):
            ax[rr, cc].set_aspect("equal", "box")

    fig.savefig(os.path.join(directories["plot_compare"], f"plot_{i:04d}.png"))
    plt.close()


if __name__ == "__main__":
    # parse args
    args = parse_arguments()

    # ====  Parameters ======================
    parameters = {
        # parameters for problem
        "problem": "ring_monitor_test",
        "lc": args.lc,
        # parameters for mesh def
        "n_samples": args.n_samples,
        "data_type": args.field_type,
        "scheme": args.boundary_scheme,
        "mesh_type": int(args.mesh_type),
        # parameters for domain scale
        "scale_x": 1,
        "scale_y": 1,
        # parameters for data split
        "p_train": 0.75,
        "p_test": 0.15,
        "p_val": 0.1,
    }

    # Set random seed
    random.seed(args.rand_seed)

    # ====  Setup Directories ======================
    problem_specific_dir = "lc={}_n={}_{}_{}_meshtype_{}".format(
        parameters["lc"],
        parameters["n_samples"],
        parameters["data_type"],
        parameters["scheme"],
        parameters["mesh_type"],
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
    i = 0
    while i < parameters["n_samples"]:
        try:
            print(f"Generating Sample: {i}")

            # create dataset
            process_features(parameters, directories)
            i += 1

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
