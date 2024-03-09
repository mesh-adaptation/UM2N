import os
import pickle
import glob
import yaml
import matplotlib.pyplot as plt
import firedrake as fd

# model_names = ["M2N", "M2N", "M2T", "M2T"]
# run_ids = ["jetaq10f", "dglbbrdq", "m9fqgqnb", "boj2eks9"]
# run_id_model_mapping = {
#     "jetaq10f": "M2N",
#     "dglbbrdq": "M2N-en",
#     "m9fqgqnb": "M2T-w-edge",
#     "boj2eks9": "M2T-n-edge",
# }
# trained_epoch = 999
# problem_type = "swirl_square"
# dataset_name = "sigma_0.017_alpha_1.5_r0_0.2_x0_0.25_y0_0.25_lc_0.028_ngrid_35_interval_5_meshtype_6_smooth_15"


# model_names = ["M2N", "M2N", "MRN"]
model_names = ["M2N", "M2N"]
run_ids = ["cyzk2mna", "u4uxcz1e"]
run_id_model_mapping = {
    "cyzk2mna": "M2N",
    "u4uxcz1e": "M2N-en",
    # "99zrohiu": "MRN",
}
trained_epoch = 999
problem_type = "helmholtz_square"
dataset_path = "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6"
dataset_name = dataset_path.split("/")[-1]
result_folder = f"./compare_output/{dataset_name}"
os.makedirs(result_folder, exist_ok=True)
is_generating_video_for_all = False
fps = 20

info_dict = {}
info_dict["run_ids"] = run_ids
info_dict["names"] = run_id_model_mapping
info_dict["epoch"] = trained_epoch
info_dict["problem_type"] = problem_type
info_dict["dataset_name"] = dataset_name
info_dict["dataset_path"] = dataset_path
# Write the dictionary to a YAML file
with open(f"{result_folder}/models_info" + ".yaml", "w") as file:
    yaml.dump(info_dict, file, default_flow_style=False)

num_vis = 5
rows = 3
cols = 3 + len(run_ids)
for n_v in range(num_vis):
    print(f"=== Visualizing number {n_v} of {dataset_name} ===")
    # Load mesh for visualization
    mesh_og = fd.Mesh(os.path.join(dataset_path, "mesh", f"mesh_{n_v:04d}.msh"))
    mesh_MA = fd.Mesh(os.path.join(dataset_path, "mesh", f"mesh_{n_v:04d}.msh"))
    mesh_fine = fd.Mesh(os.path.join(dataset_path, "mesh_fine", f"mesh_{n_v:04d}.msh"))
    mesh_model = fd.Mesh(os.path.join(dataset_path, "mesh", f"mesh_{n_v:04d}.msh"))

    model_function_space = fd.FunctionSpace(mesh_model, "CG", 1)
    high_res_function_space = fd.FunctionSpace(mesh_fine, "CG", 1)

    u_og = fd.Function(model_function_space)
    u_ma = fd.Function(model_function_space)
    u_model = fd.Function(model_function_space)
    monitor_values = fd.Function(model_function_space)

    # u exact lives in high res function space
    u_exact = fd.Function(high_res_function_space)
    error_map_original = fd.Function(high_res_function_space)
    error_map_ma = fd.Function(high_res_function_space)
    error_map_model = fd.Function(high_res_function_space)

    fig, ax = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5), layout="compressed"
    )
    cnt = 0

    plot_data_dicts = {}
    for model_name, run_id in zip(model_names, run_ids):
        show_name = run_id_model_mapping[run_id]
        print(f"model name: {show_name}, run id: {run_id}")
        eval_ret_path = f"./eval/{model_name}_{trained_epoch}_{run_id}/{problem_type}/{dataset_name}"
        # print(eval_ret_path)
        eval_exp_path = sorted(glob.glob(f"{eval_ret_path}/*"))[-1]
        eval_plot_data_path = os.path.join(eval_exp_path, "plot_data")
        eval_plot_more_path = os.path.join(eval_exp_path, "plot_more")

        if is_generating_video_for_all:
            # Generate videos
            chdir_command = f"cd {eval_plot_more_path}"
            video_command = f"ti video -f {fps}"
            os.system(f"{chdir_command} && {video_command}")

        eval_plot_data_files = sorted(glob.glob(f"{eval_plot_data_path}/*"))

        with open(eval_plot_data_files[n_v], "rb") as f:
            plot_data_dict = pickle.load(f)
        plot_data_dicts[run_id] = plot_data_dict

    error_v_max = None
    solution_v_max = None
    solution_v_min = None

    for model_name, run_id in zip(model_names, run_ids):
        plot_data_dict = plot_data_dicts[run_id]
        if not error_v_max:
            error_v_max = plot_data_dict["error_v_max"]
        else:
            error_v_max = max(error_v_max, plot_data_dict["error_v_max"])

        if not solution_v_max:
            solution_v_max = plot_data_dict["u_v_max"]
        else:
            solution_v_max = max(solution_v_max, plot_data_dict["u_v_max"])

        if not solution_v_min:
            solution_v_min = plot_data_dict["u_v_min"]
        else:
            solution_v_min = max(solution_v_min, plot_data_dict["u_v_min"])

    # Visualize all
    # Load all data first because we need a fair vmax and vmin
    for model_name, run_id in zip(model_names, run_ids):
        plot_data_dict = plot_data_dicts[run_id]

        u_exact_data = plot_data_dict["u_exact"]
        u_exact.dat.data[:] = u_exact_data

        u_og_data = plot_data_dict["u_original"]
        u_og.dat.data[:] = u_og_data

        u_ma_data = plot_data_dict["u_ma"]
        u_ma.dat.data[:] = u_ma_data

        # u_model exists if no mesh tangling
        if "u_model" in plot_data_dict:
            u_model_data = plot_data_dict["u_model"]
            u_model.dat.data[:] = u_model_data

        error_map_original_data = plot_data_dict["error_map_original"]
        error_map_original.dat.data[:] = error_map_original_data

        error_map_ma_data = plot_data_dict["error_map_ma"]
        error_map_ma.dat.data[:] = error_map_ma_data

        if "error_map_model" in plot_data_dict:
            error_map_model_data = plot_data_dict["error_map_model"]
            error_map_model.dat.data[:] = error_map_model_data

        error_og_mesh = plot_data_dict["error_norm_original"]
        error_ma_mesh = plot_data_dict["error_norm_ma"]

        if "error_norm_model" in plot_data_dict:
            error_model_mesh = plot_data_dict["error_norm_model"]
        else:
            error_model_mesh = -1

        cmap = "seismic"
        show_name = run_id_model_mapping[run_id]

        mesh_model.coordinates.dat.data[:] = plot_data_dict["mesh_model"]
        # Adapted mesh (Model)
        fd.triplot(mesh_model, axes=ax[0, 3 + cnt])
        ax[0, 3 + cnt].set_title(f"Adapted Mesh ({show_name})")
        # Solution on adapted mesh (Model)
        cb = fd.tripcolor(
            u_model,
            cmap=cmap,
            vmax=solution_v_max,
            vmin=solution_v_min,
            axes=ax[1, 3 + cnt],
        )
        ax[1, 3 + cnt].set_title(f"Solution on Adapted Mesh ({show_name})")
        plt.colorbar(cb)
        # Error map (Model)
        cb = fd.tripcolor(
            error_map_model,
            cmap=cmap,
            vmax=error_v_max,
            vmin=-error_v_max,
            axes=ax[2, 3 + cnt],
        )
        ax[2, 3 + cnt].set_title(
            f"Error (u-u_exact) {model_name}| L2 Norm: {error_model_mesh:.5f} | {(error_og_mesh-error_model_mesh)/error_og_mesh*100:.2f}%"
        )
        plt.colorbar(cb)

        cnt += 1

    # Fill the first three columns
    plot_data_dict = plot_data_dicts[run_ids[0]]

    mesh_ma_data = plot_data_dict["mesh_ma"]
    mesh_MA.coordinates.dat.data[:] = mesh_ma_data
    monitor_values_data = plot_data_dict["monitor_values"]
    monitor_values.dat.data[:] = monitor_values_data

    err_map_orignal_data = plot_data_dict["error_map_original"]
    error_map_original.dat.data[:] = err_map_orignal_data

    err_map_ma_data = plot_data_dict["error_map_ma"]
    error_map_ma.dat.data[:] = err_map_ma_data

    u_exact_data = plot_data_dict["u_exact"]
    u_exact.dat.data[:] = u_exact_data

    u_og_data = plot_data_dict["u_original"]
    u_og.dat.data[:] = u_og_data

    u_ma_data = plot_data_dict["u_ma"]
    u_ma.dat.data[:] = u_ma_data

    # High resolution mesh
    fd.triplot(mesh_fine, axes=ax[0, 0])
    ax[0, 0].set_title(f"High resolution Mesh (100 x 100)")
    # Orginal low resolution uniform mesh
    fd.triplot(mesh_og, axes=ax[0, 1])
    ax[0, 1].set_title(f"Original uniform Mesh")
    # Adapted mesh (MA)
    fd.triplot(mesh_MA, axes=ax[0, 2])
    ax[0, 2].set_title(f"Adapted Mesh (MA)")

    # Solution on high resolution mesh
    cb = fd.tripcolor(
        u_exact, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 0]
    )
    ax[1, 0].set_title(f"Solution on High Resolution (u_exact)")
    plt.colorbar(cb)
    # Solution on orginal low resolution uniform mesh
    cb = fd.tripcolor(
        u_og, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 1]
    )
    ax[1, 1].set_title(f"Solution on uniform Mesh")
    plt.colorbar(cb)
    # Solution on adapted mesh (MA)
    cb = fd.tripcolor(
        u_ma, cmap=cmap, vmax=solution_v_max, vmin=solution_v_min, axes=ax[1, 2]
    )
    ax[1, 2].set_title(f"Solution on Adapted Mesh (MA)")
    plt.colorbar(cb)

    # Monitor values
    cb = fd.tripcolor(monitor_values, cmap=cmap, axes=ax[2, 0])
    ax[2, 0].set_title(f"Monitor values")
    plt.colorbar(cb)

    # Error on orginal low resolution uniform mesh
    cb = fd.tripcolor(
        error_map_original,
        cmap=cmap,
        axes=ax[2, 1],
        vmax=error_v_max,
        vmin=-error_v_max,
    )
    ax[2, 1].set_title(f"Error (u-u_exact) uniform Mesh | L2 Norm: {error_og_mesh:.5f}")
    plt.colorbar(cb)
    # Error on adapted mesh (MA)
    cb = fd.tripcolor(
        error_map_ma,
        cmap=cmap,
        axes=ax[2, 2],
        vmax=error_v_max,
        vmin=-error_v_max,
    )
    ax[2, 2].set_title(
        f"Error (u-u_exact) MA| L2 Norm: {error_ma_mesh:.5f} | {(error_og_mesh-error_ma_mesh)/error_og_mesh*100:.2f}%"
    )
    plt.colorbar(cb)

    for rr in range(rows):
        for cc in range(cols):
            ax[rr, cc].set_aspect("equal", "box")

    fig.savefig(f"{result_folder}/compare_ret_{n_v:04d}.png")


# Generate video for compare results
chdir_command = f"cd {result_folder}"
video_command = f"ti video -f {fps}"
os.system(f"{chdir_command} && {video_command}")
