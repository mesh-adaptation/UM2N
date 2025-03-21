import datetime
import glob
import os
import pickle
import time
from types import SimpleNamespace

import firedrake as fd
import matplotlib.pyplot as plt  # noqa
import pandas as pd
import torch
import wandb
from torch_geometric.loader import DataLoader

import UM2N
from UM2N.model.train_util import model_forward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sample_param_of_nu_generalization_by_idx_train(idx_in):
    gauss_list_ = []
    if idx_in == 1:
        param_ = {"cx": 0.225, "cy": 0.5, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 2:
        param_ = {"cx": 0.225, "cy": 0.5, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 3:
        param_ = {"cx": 0.225, "cy": 0.5, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 4:
        shift_ = 0.15
        param_ = {"cx": 0.3, "cy": 0.5 - shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.15, "cy": 0.5 + shift_, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 5:
        shift_ = 0.15
        param_ = {"cx": 0.3, "cy": 0.5 - shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.15, "cy": 0.5 + shift_, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 6:
        shift_ = 0.15
        param_ = {"cx": 0.3, "cy": 0.5 - shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.15, "cy": 0.5 + shift_, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 7:
        shift_ = 0.2
        param_ = {"cx": 0.3, "cy": 0.5 + shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.3, "cy": 0.5 - shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.15, "cy": 0.5, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 8:
        shift_ = 0.2
        param_ = {"cx": 0.3, "cy": 0.5 + shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.3, "cy": 0.5 - shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.15, "cy": 0.5, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 9:
        shift_ = 0.2
        param_ = {"cx": 0.3, "cy": 0.5 + shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.3, "cy": 0.5 - shift_, "w": 0.01}
        gauss_list_.append(param_)
        param_ = {"cx": 0.15, "cy": 0.5, "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    return gauss_list_, nu_


def init_dir(config, run_id, epoch, ds_root, problem_type, domain):
    """
    Make dir for evaluation. All evaluation files will be stored
    under the dir created.
    """
    print("\t## Making directories to store files")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(project_dir, "eval")
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ds_name = ds_root.split("/")[-1]
    experiment_dir = os.path.join(
        eval_dir,
        config.model_used + "_" + str(epoch) + "_" + run_id,
        problem_type + "_" + domain,
        f"{ds_name}",
        now,
    )
    UM2N.mkdir_if_not_exist(experiment_dir)
    print("\t## Make eval dir done\n")
    return experiment_dir


def load_dataset(
    config, ds_root, tar_folder, use_run_time_cluster=False, use_cluster=False
):
    """
    prepare datasets to use.

    Args:
        config: config for the model.
        ds_root: root path to the dataset to be loaded.
        tar_folder: name of the folder containing input samples.
        use_run_time_cluster: flag to turn on cluster on-the-fly generation.
        use_cluster: flag controling whether to use cluset in training.
    """
    dataset_path = os.path.join(ds_root, tar_folder)
    dataset = UM2N.MeshDataset(
        dataset_path,
        transform=UM2N.normalise if UM2N.normalise else None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        load_analytical=True,
        use_run_time_cluster=use_run_time_cluster,
        use_cluster=use_cluster,
        r=0.35,
        M=25,
        dist_weight=False,
        add_nei=True,
    )
    return dataset


def load_model(config, epoch, experiment_dir):
    """
    Load Model to evaluate, prepare datasets to use.
    Also Make dir for evaluation. All evaluation files will be stored
    under the dir created.

    Args:
        config (SimpleNamespace): config for the model run
        ds_root (str): path to root data folder.
        epoch: number of epoch the model been loaded from.

    Returns:
        model: the model loaded from wandb api.
        dataset: the dataset used to train the model.
        eval_dir: the path of root dir of evaluation files.
    """

    target_file_name = "model_{}.pth".format(epoch)
    model_file = None
    for file in run.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(root=experiment_dir, replace=True)
            target_file_name = file.name
    assert model_file is not None, "Model file not found"
    model = None
    if config.model_used == "M2N":
        model = UM2N.M2N(
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
            deform_in_c=config.num_deform_in,
        )
    elif config.model_used == "MRN":
        model = UM2N.MRN(
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
            deform_in_c=config.num_deform_in,
            num_loop=config.num_deformer_loop,
        )
    elif config.model_used == "MRT" or config.model_used == "MRTransformer":
        model = UM2N.MRTransformer(
            num_transformer_in=config.num_transformer_in,
            num_transformer_out=config.num_transformer_out,
            num_transformer_embed_dim=config.num_transformer_embed_dim,
            num_transformer_heads=config.num_transformer_heads,
            num_transformer_layers=config.num_transformer_layers,
            transformer_training_mask=config.transformer_training_mask,
            transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,  # noqa
            transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,  # noqa
            deform_in_c=config.num_deform_in,
            deform_out_type=config.deform_out_type,
            num_loop=config.num_deformer_loop,
            device=device,
        )
    elif config.model_used == "M2T":
        model = UM2N.M2T(
            num_transformer_in=config.num_transformer_in,
            num_transformer_out=config.num_transformer_out,
            num_transformer_embed_dim=config.num_transformer_embed_dim,
            num_transformer_heads=config.num_transformer_heads,
            num_transformer_layers=config.num_transformer_layers,
            transformer_training_mask=config.transformer_training_mask,
            transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,  # noqa
            transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,  # noqa
            deform_in_c=config.num_deform_in,
            local_feature_dim_in=config.num_lfe_in,
            deform_out_type=config.deform_out_type,
            num_loop=config.num_deformer_loop,
            device=device,
        )
    elif config.model_used == "M2N_T":
        model = UM2N.M2N_T(
            deform_in_c=config.num_deform_in,
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
        )
    else:
        print("Model not found")
    model_file_path = os.path.join(experiment_dir, target_file_name)
    model = UM2N.load_model(model, model_file_path)
    return model


def get_log_og(log_path, idx):
    """
    Read log file from dataset log dir and return value in it
    """
    df = pd.read_csv(os.path.join(log_path, f"log_{idx:04d}.csv"))
    return {
        "error_og": df["error_og"][0],
        "error_adapt": df["error_adapt"][0],
        "time": df["time"][0],
    }


def get_problem_type(ds_root):
    domain = None
    print("ds_root", ds_root)
    ds_type = ds_root.split("/")[-2]
    meshtype = int(ds_root.split("/")[-3].split("_")[-1])
    # meshtype = 6
    problem_list = ds_type.split("_")
    problem_type = problem_list[0]
    if len(problem_list) == 2:
        if problem_list[-1] == "poly":
            domain = "poly"
    else:
        domain = "square"
    return problem_type, domain, meshtype


def benchmark_model(model, dataset, eval_dir, ds_root, start_idx=0, num_samples=100):
    """ "
    benchmark the performance of a model on a specific data
    set.

    Args:
        data_path (str): the path to the datasets root to evaluate.
        tar_folder (str): the target folder name containing the input samples.
        eval_dir (str): path to the evalution dir a.k.a. data root.
        num_samples (int): number of samples used to do the evaluation.
    """
    # domain = None
    # ds_type = ds_root.split('/')[-2]
    # problem_list = ds_type.split('_')
    # problem_type = problem_list[0]
    # if (len(problem_list) == 2):
    #     if (problem_list[-1] == 'poly'):
    #         domain = 'poly'
    # else:
    #     domain = 'square'
    # print('problem type: ', problem_type, domain)
    problem_type, domain, meshtype = get_problem_type(ds_root=ds_root)
    print(f"problem type: {problem_type}, domain: {domain}, meshtype: {meshtype}")

    if problem_type == "helmholtz" or problem_type == "poisson":
        n_grid = None
        if int(meshtype) == 0:
            n_grid = int(ds_root.split("/")[-1].split(">")[-2][-2:])
            print(f"meshtype {meshtype}, n_grid: {n_grid}")
        mesh = None
        mesh_fine = None

        log_dir = os.path.join(eval_dir, "log")
        plot_dir = os.path.join(eval_dir, "plot")
        plot_more_dir = os.path.join(eval_dir, "plot_more")
        plot_data_dir = os.path.join(eval_dir, "plot_data")
        print("log_dir issss", log_dir)
        UM2N.mkdir_if_not_exist(log_dir)
        UM2N.mkdir_if_not_exist(plot_dir)
        UM2N.mkdir_if_not_exist(plot_more_dir)
        UM2N.mkdir_if_not_exist(plot_data_dir)

        model = model.to(device)
        total_num = len(dataset)
        print(f"Total num: {len(dataset)}")
        for idx in range(start_idx, min(total_num, start_idx + num_samples)):
            # model inference stage
            # print(len(dataset))
            print("IDX", idx)
            sample = next(iter(DataLoader([dataset[idx]], batch_size=1, shuffle=False)))
            model.eval()
            bs = 1
            sample = sample.to(device)
            with torch.no_grad():
                start = time.perf_counter()
                if config.model_used == "MRTransformer" or config.model_used == "M2T":
                    data = sample
                    (
                        output_coord,
                        output,
                        out_monitor,
                        phix,
                        phiy,
                        mesh_query_x_all,
                        mesh_query_y_all,
                    ) = model_forward(
                        bs,
                        data,
                        model,
                        use_add_random_query=config.use_add_random_query,
                    )
                    out = output_coord
                elif config.model_used == "M2N" or config.model_used == "M2N_T":
                    # print(sample)
                    out = model(sample)
                elif config.model_used == "MRN":
                    out = model(sample)
                else:
                    raise Exception(f"model {config.model_used} not implemented.")
                end = time.perf_counter()
                dur_ms = (end - start) * 1000
            temp_time_consumption = dur_ms
            temp_tangled_elem = UM2N.get_sample_tangle(out, sample.y, sample.face)
            temp_loss = 1000 * torch.nn.L1Loss()(out, sample.y)
            # define mesh & fine mesh for comparison
            if domain == "square":
                if n_grid is None:
                    mesh = fd.Mesh(os.path.join(ds_root, "mesh", f"mesh_{idx:04d}.msh"))
                    mesh_MA = fd.Mesh(
                        os.path.join(ds_root, "mesh", f"mesh_{idx:04d}.msh")
                    )
                    mesh_fine = fd.Mesh(
                        os.path.join(ds_root, "mesh_fine", f"mesh_{idx:04d}.msh")
                    )
                    mesh_model = fd.Mesh(
                        os.path.join(ds_root, "mesh", f"mesh_{idx:04d}.msh")
                    )
                else:
                    mesh = fd.UnitSquareMesh(n_grid, n_grid)
                    mesh_MA = fd.UnitSquareMesh(n_grid, n_grid)
                    mesh_fine = fd.UnitSquareMesh(100, 100)
                    mesh_model = fd.UnitSquareMesh(n_grid, n_grid)
            elif domain == "poly":
                mesh = fd.Mesh(os.path.join(ds_root, "mesh", f"mesh_{idx:04d}.msh"))
                mesh_MA = fd.Mesh(os.path.join(ds_root, "mesh", f"mesh_{idx:04d}.msh"))
                mesh_fine = fd.Mesh(
                    os.path.join(ds_root, "mesh_fine", f"mesh_{idx:04d}.msh")
                )
                mesh_model = fd.Mesh(
                    os.path.join(ds_root, "mesh", f"mesh_{idx:04d}.msh")
                )
            mesh_model.coordinates.dat.data[:] = out.detach().cpu().numpy()

            compare_res = UM2N.compare_error(
                sample,
                mesh,
                mesh_fine,
                mesh_model,
                mesh_MA,
                temp_tangled_elem,
                model_name=config.model_used,
                problem_type=problem_type,
            )
            temp_error_model = compare_res["error_model_mesh"]
            temp_error_og = compare_res["error_og_mesh"]
            temp_error_ma = compare_res["error_ma_mesh"]

            compare_res["plot_data_dict"]["deform_loss"] = temp_loss.cpu().numpy()
            plot_data_dict = compare_res["plot_data_dict"]

            # Save plot data
            with open(
                os.path.join(plot_data_dir, f"plot_data_{idx:04d}.pkl"), "wb"
            ) as p:
                pickle.dump(plot_data_dict, p)

            plot_more = compare_res["plot_more"]
            plot_more.savefig(os.path.join(plot_more_dir, f"plot_{idx:04d}.png"))
            plt.close()

            if int(meshtype) != 0:
                log_og = get_log_og(os.path.join(ds_root, "log"), idx)
                log_error_og = log_og["error_og"]
                log_error_ma = log_og["error_adapt"]
                log_time = log_og["time"]
            else:
                # TODO: no corresponding records in old dataset
                log_error_og = 0.0
                log_error_ma = 0.0
                log_time = 0.0

            error_reduction_MA = (temp_error_og - temp_error_ma) / temp_error_og
            error_reduction_model = (temp_error_og - temp_error_model) / temp_error_og
            # log file
            log_df = pd.DataFrame(
                {
                    "deform_loss": temp_loss.cpu().numpy(),
                    "tangled_element": temp_tangled_elem,
                    "error_og": temp_error_og,
                    "error_model": temp_error_model,
                    "error_ma": temp_error_ma,
                    "error_log_og": log_error_og,
                    "error_log_ma": log_error_ma,
                    "error_reduction_MA": (temp_error_og - temp_error_ma)
                    / temp_error_og,  # noqa
                    "error_reduction_model": (temp_error_og - temp_error_model)
                    / temp_error_og,  # noqa
                    "time_consumption_model": temp_time_consumption,
                    "time_consumption_MA": log_time,
                    "acceration_ratio": log_time / temp_time_consumption,
                },
                index=[0],
            )
            log_df.to_csv(os.path.join(log_dir, f"log_{idx:04d}.csv"))

            fig = UM2N.plot_mesh_compare_benchmark(
                out.cpu(),
                sample.y.cpu(),
                sample.face.cpu(),
                deform_loss=temp_loss,
                pde_loss_model=temp_error_model,
                pde_loss_reduction_model=error_reduction_model,
                pde_loss_MA=temp_error_ma,
                pde_loss_reduction_MA=error_reduction_MA,
                tangle=temp_tangled_elem,
            )
            fig.savefig(os.path.join(plot_dir, f"plot_{idx:04d}.png"))
            plt.close()

    elif problem_type == "swirl":
        n_grid = None
        if int(meshtype) == 0:
            n_grid = int(ds_root.split("/")[-1].split("ngrid_")[-1][:2])
            print(f"meshtype {meshtype}, n_grid: {n_grid}")
        mesh = None
        mesh_fine = None

        # Readin params for a specific swirl problem
        info_df = pd.read_csv(os.path.join(ds_root, "info.csv"))
        sigma = info_df["sigma"][0]
        alpha = info_df["alpha"][0]
        r_0 = info_df["r_0"][0]
        x_0 = info_df["x_0"][0]
        y_0 = info_df["y_0"][0]
        T = info_df["T"][0]
        n_step = info_df["n_step"][0]
        dt = info_df["dt"][0]
        save_interval = info_df["save_interval"][0]

        if n_grid is None:
            mesh = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
            # mesh_coarse = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
            mesh_ma = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
            mesh_model = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
            mesh_fine = fd.Mesh(os.path.join(ds_root, "mesh_fine", "mesh.msh"))
        else:
            mesh = fd.UnitSquareMesh(n_grid, n_grid)
            # mesh_coarse = fd.UnitSquareMesh(n_grid, n_grid)
            mesh_ma = fd.UnitSquareMesh(n_grid, n_grid)
            mesh_model = fd.UnitSquareMesh(n_grid, n_grid)
            mesh_fine = fd.UnitSquareMesh(100, 100)

        # solver defination
        swril_solver = UM2N.SwirlSolver(
            mesh,
            mesh_fine,
            mesh_ma,
            mesh_model,
            dataset=dataset,
            sigma=sigma,
            alpha=alpha,
            r_0=r_0,
            x_0=x_0,
            y_0=y_0,
            save_interval=save_interval,
            T=T,
            dt=dt,
            n_step=n_step,
        )

        # evaluator = UM2N.SwirlEvaluator(
        #     mesh,
        #     mesh_coarse,
        #     mesh_fine,
        #     mesh_new,
        #     mesh_model,
        #     dataset,
        #     model,
        #     eval_dir,
        #     ds_root,
        #     device=device,
        #     sigma=sigma,
        #     alpha=alpha,
        #     r_0=r_0,
        #     x_0=x_0,
        #     y_0=y_0,
        #     T=T,
        #     n_step=n_step,
        #     model_used=config.model_used,
        #     num_samples_to_eval=num_samples,
        # )

        # evaluator.make_log_dir()
        # evaluator.make_plot_dir()
        # evaluator.make_plot_more_dir()
        # evaluator.make_plot_data_dir()

        model_name = config.model_used
        if config.model_used == "MRTransform":
            model_name = "M2T"
        eval_res = swril_solver.eval_problem(
            model, ds_root=ds_root, eval_dir=eval_dir, model_name=model_name
        )  # noqa

    elif problem_type == "burgers":
        # Select params to generate burgers bump
        case_idxs = [1]
        for idx in case_idxs:
            gaussian_list, nu = get_sample_param_of_nu_generalization_by_idx_train(idx)  # noqa
            mesh = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
            mesh_new = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
            mesh_fine = fd.Mesh(os.path.join(ds_root, "mesh_fine", "mesh.msh"))

            evaluator = UM2N.BurgersEvaluator(
                mesh,
                mesh_fine,
                mesh_new,
                dataset,
                model,
                eval_dir,
                ds_root,
                idx,
                device=device,
                gauss_list=gaussian_list,
                nu=nu,
                model_used=config.model_used,
            )

            evaluator.make_log_dir()
            evaluator.make_plot_dir()
            evaluator.make_plot_more_dir()
            evaluator.make_plot_data_dir()

            eval_res = evaluator.eval_problem()  # noqa


def write_sumo(eval_dir, ds_root):
    log_dir = os.path.join(eval_dir, "log")
    file_path = os.path.join(log_dir, "log*.csv")
    log_files = glob.glob(file_path)
    log_files = sorted(log_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    deform_loss = 0
    error_MA = 0
    error_model = 0
    error_og = 0
    time_MA = 0
    num_tangle = 0
    time_model = 0
    pass_count = 0
    fail_count = 0
    total_count = 0

    error_MA_all = []
    error_model_all = []
    error_og_all = []

    error_reduction_MA = 0
    error_reduction_model = 0
    error_reduction_MA_all = []
    error_reduction_model_all = []

    for file_names in log_files:
        total_count += 1
        log_df = pd.read_csv(file_names)
        # print(log_df['tangled_element'][0], log_df['tangled_element'][0] == 0)
        if log_df["tangled_element"][0] == 0:
            pass_count += 1
            deform_loss += log_df["deform_loss"][0]
            error_og += log_df["error_og"][0]
            error_MA += log_df["error_ma"][0]
            error_model += log_df["error_model"][0]

            error_MA_all.append(log_df["error_ma"][0])
            error_model_all.append(log_df["error_model"][0])
            error_og_all.append(log_df["error_og"][0])

            error_reduction_MA += log_df["error_reduction_MA"][0]
            error_reduction_MA_all.append(log_df["error_reduction_MA"][0])

            error_reduction_model += log_df["error_reduction_model"][0]
            error_reduction_model_all.append(log_df["error_reduction_model"][0])

            time_MA += log_df["time_consumption_MA"][0]
            time_model += log_df["time_consumption_model"][0]
        else:
            fail_count += 1
            num_tangle += log_df["tangled_element"][0]
    print(f"passed num: {pass_count}, failed num: {fail_count}")
    if pass_count == 0:
        pass_count = 1
    sumo_df = pd.DataFrame(
        {
            # 'error_reduction_MA': (error_og - error_MA) / error_og,
            # 'error_reduction_model': (error_og - error_model) / error_og,'
            "error_reduction_MA": error_reduction_MA / pass_count,
            "error_reduction_model": error_reduction_model / pass_count,
            "deform_loss": deform_loss / pass_count,
            "time_MA": time_MA,
            "time_model": time_model,
            "acceleration_ratio": time_MA / time_model if time_model != 0 else 0,
            "tangle_total": num_tangle,
            "TEpM(tangled Elements per Mesh)": num_tangle / total_count,
            "failed_case": fail_count,
            "total_case": total_count,
            "dataset_path": ds_root,
        },
        index=[0],
    )
    print(
        f"[summary] error_reduction_MA: {sumo_df['error_reduction_MA'][0]}, error_reduction_model: {sumo_df['error_reduction_model'][0]}, deform loss: {sumo_df['deform_loss'][0]}"
    )
    print(" ")
    summary_save_path = os.path.join(eval_dir)
    sumo_df.to_csv(os.path.join(summary_save_path, "sumo.csv"))

    # Visualize the error reduction
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(
        [x for x in range(len(error_reduction_MA_all))],
        error_reduction_MA_all,
        label="Error reduction (MA)",
    )
    ax[0].plot(
        [x for x in range(len(error_reduction_model_all))],
        error_reduction_model_all,
        label="Error reduction (model)",
    )
    ax[0].legend()

    ax[1].plot(
        [x for x in range(len(error_MA_all))], error_MA_all, label="PDE error (MA)"
    )
    ax[1].plot(
        [x for x in range(len(error_model_all))],
        error_model_all,
        label="PDE error (Model)",
    )
    ax[1].plot(
        [x for x in range(len(error_og_all))],
        error_og_all,
        label="PDE error (uniform)",
        color="k",
    )
    ax[1].legend()

    fig_title = ds_root.split("/")[-1]
    fig.suptitle(f"{fig_title}", fontsize=16)
    fig.savefig(os.path.join(summary_save_path, "error_reduction_sumo.png"))

    big_df_res = UM2N.write_stat(eval_dir)
    big_df_res["fig"].savefig(os.path.join(summary_save_path, "error_hist.png"))  # noqa
    big_df_res["df"].to_csv(os.path.join(summary_save_path, "all_info.csv"))


if __name__ == "__main__":
    entity = "mz-team"
    # entity = 'w-chunyang'
    project_name = "warpmesh"
    epoch = 999

    run_id = "8ndi2teh"  # semi-supervised phi grad
    run_id = "bzlj9vcl"  # semi-supervised 111
    run_id = "x9woqsnn"  # supervised phi grad
    run_id = "7py7k3ah"  # fine tune on helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2
    run_id = "uka7cidv"  # fine tune on helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2, freeze deformer
    run_id = "81b3gh8y"  # fine tune on supervised helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2, freeze deformer
    run_id = "0ejnq1mt"  # fine tune on supervised helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2, freeze deformer
    run_id = "dnolwyeb"  # fine tune on supervised helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2, freeze transformer
    # run_id = 'bxrlm3dl' # fine tune on supervised helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2, no freeze
    run_id = "eanjdljm"  # fine tune with to monitor on unsupervised helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2, freeze both
    run_id = "cbzxfq1o"  # semi-supervised from old dataset with to monitor

    run_id = "irjq8z0r"  # supervised from old dataset with to monitor

    run_id = (
        "b64qp0b3"  # supervised from old dataset with to monitor with interpolation
    )

    # run_id = 'yn3aaiwi' # mesh query semi 111
    run_id = "pk66tmjj"  # mesh query semi 111 50 smaples

    # run_id = '1cf7cu3d' # mesh query purely supervised

    run_id = "0l8ujpdr"  # mesh query semi 111, old dataset
    run_id = "hmgwx4ju"  # mesh query semi 011 (purely supervised), old dataset
    run_id = "tlvacka0"  # 1 0 0, pure unsupervised '0l8ujpdr' fine tune on './data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6'
    run_id = "989eagtl"  # 1 1 1, semi unsupervised '0l8ujpdr' fine tune on './data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6'
    run_id = "knjfc14i"  # 1 1 1, semi unsupervised '0l8ujpdr' fine tune on './data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6'

    run_id = "cbey3q32"  # 1 1 1, semi unsupervised '0l8ujpdr' fine tune on './data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6', freeze transformer

    run_id = "boe36e11"  # 0 1 1, pure supervised '0l8ujpdr' fine tune on './data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6'

    run_id = "6oel4b5v"  # enhanced with random sampling

    run_id = "28ihwvfg"  # 6oel4b5v continue train

    run_id = "2x84suu1"  # with random sampling in deformer, semi-trained on meshtype 6 50 samples

    run_id_pi_m2t = (
        "4a1p7ekj"  # trained with 600 samples, semi with random sampling query
    )
    run_id_m2t = "99zrohiu"  # trained with 600 samples, purely supervised

    # run_id = "6fxictgr" # M2N baseline, trained on 600 samples meshtype 2 helmholtz (wrong)

    run_id_mrn = "7qxbs9nt"  # MRN
    run_id_mrn_area_loss = "xwds86sw"  # MRN area loss
    run_id_mrn_hessian_norm = "8vftl7it"  # MRN hessian norm
    run_id_mrn_area_loss_hessian_norm = "ilhhvshe"  # MRN area loss, hessian norm

    run_id_m2n = "cyzk2mna"  # M2N
    run_id_m2n_area_loss = "xfjd6pdm"  # M2N area loss
    run_id_m2n_hessian_norm = "1cu4qw9u"  # M2N hessian norm
    run_id_m2n_area_loss_hessian_norm = "u4uxcz1e"  # M2N area loss hessian norm

    # run_ids = ['8ndi2teh', 'x9woqsnn']
    # run_ids = ['0l8ujpdr', 'hmgwx4ju', '8ndi2teh']

    # run_ids = [run_id_mrn, run_id_mrn_area_loss, run_id_mrn_hessian_norm, run_id_mrn_area_loss_hessian_norm,
    #            run_id_m2n, run_id_m2n_area_loss, run_id_m2n_hessian_norm, run_id_m2n_area_loss_hessian_norm]
    # run_ids = [run_id_m2n_area_loss_hessian_norm, run_id_mrn_area_loss_hessian_norm]
    # run_ids = [run_id_m2n_area_loss_hessian_norm, run_id_mrn_area_loss_hessian_norm, run_id_m2t, run_id_pi_m2t]
    run_ids_old_benchmark = [
        # run_id_pi_m2t,
        run_id_m2n_area_loss_hessian_norm,
        run_id_m2n,
        run_id_m2t,
    ]

    ##########################################
    ## Mini set Area (train on old dataset) ##
    ##########################################
    run_id_m2n_miniset = "cy82mqh2"
    run_id_m2n_enhance_miniset = "u34s941u"
    run_id_MRT_miniset = "oasvyxn1"
    run_id_PIMRT_miniset_old = "u5ni7dtm"
    run_id_PIMRT_miniset = "3xe6n0nz"

    run_ids_miniset_old = [
        run_id_m2n_miniset,
        run_id_m2n_enhance_miniset,
        run_id_MRT_miniset,
        run_id_PIMRT_miniset,
    ]

    ##########################################
    ## Mini set Area (train on new dataset) ##
    ##########################################
    run_id_m2n_miniset_new = "jetaq10f"
    run_id_m2n_enhance_miniset_new = "dglbbrdq"
    run_id_MRT_miniset_new = "rud1gsge"
    run_id_PIMRT_miniset_new = "qvnnvt65"
    run_id_M2T_miniset_new = "m9fqgqnb"  # With edge
    run_id_M2T_miniset_new = "boj2eks9"  # No edge

    run_ids_miniset_new = [
        run_id_m2n_miniset_new,
        run_id_m2n_enhance_miniset_new,
        # run_id_M2T_miniset_new,
        # run_id_MRT_miniset_new,
        # run_id_PIMRT_miniset_new,
    ]

    ##########################################
    ## Large set Area (train on new dataset) ##
    ##########################################
    run_id_m2n_largeset = "b4fx4wz5"
    run_id_m2n_enhance_largeset = "u6zpxaoz"
    run_id_MRT_largeset = "kgr0nicn"
    run_id_PIMRT_largeset = "6wkt13wp"
    run_id_M2T_largeset = "gywsmly9"  # pre-train using "rud1gsge"
    run_id_M2T_largeset = "ig1np6kx"  # pre-train using "kgr0nicn", this is the good one
    run_id_M2T_largeset = "72esorfm"  # pre-train using "99zrohiu"

    run_ids_largeset = [
        run_id_m2n_largeset,
        run_id_m2n_enhance_largeset,
        run_id_M2T_largeset,
        run_id_MRT_largeset,
        run_id_PIMRT_largeset,
    ]

    ds_root_helmholtz = [
        "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2",
        "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6",
        "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=100_aniso_full_meshtype_6",
        "./data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<15x15>_n=100_aniso_full",
        "./data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<20x20>_n=100_aniso_full",
        "./data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<35x35>_n=100_aniso_full",
    ]

    ds_root_helmholtz = [
        # "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=300_aniso_full_meshtype_6"
        "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=101_aniso_full_meshtype_6"
        # "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=5_aniso_full_meshtype_6",
        # "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6",
        # "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=100_aniso_full_meshtype_6",
        # "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2",
        # "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=100_aniso_full_meshtype_2",
        # "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=300_aniso_full_meshtype_2",
    ]

    # ds_root_helmholtz = [
    #     # "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=5_aniso_full_meshtype_6",
    #     "./data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=50_aniso_full_algo_6"
    #     # "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=300_aniso_full_meshtype_2",
    # ]

    ds_root_swirl = [
        # "./data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.5_y0_0.5_lc_0.028_ngrid_35_interval_5_meshtype_6_smooth_15",
        "./data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.3_y0_0.3_lc_0.028_ngrid_35_interval_10_meshtype_6_smooth_15",
        # "./data/dataset_meshtype_2/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.25_y0_0.25_lc_0.028_ngrid_35_interval_5_meshtype_2_smooth_15",
        # "./data/dataset_meshtype_0/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.25_y0_0.25_lc_0.028_ngrid_35_interval_5_meshtype_0_smooth_15",
    ]

    ds_root_swirl = [
        # "./data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.5_y0_0.5_lc_0.028_ngrid_35_interval_5_meshtype_6_smooth_15",
        "./data/dataset_meshtype_6/swirl/sigma_0.017_alpha_0.9_r0_0.2_x0_0.3_y0_0.3_lc_0.028_ngrid_35_interval_10_meshtype_6_smooth_10",
        # "./data/dataset_meshtype_2/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.25_y0_0.25_lc_0.028_ngrid_35_interval_5_meshtype_2_smooth_15",
        # "./data/dataset_meshtype_0/swirl/sigma_0.017_alpha_1.5_r0_0.2_x0_0.25_y0_0.25_lc_0.028_ngrid_35_interval_5_meshtype_0_smooth_15",
    ]

    ds_root_burgers = [
        "./data/dataset_meshtype_2/burgers/lc=0.05_ngrid_20_n=5_iso_pad_meshtype_2",
        "./data/dataset_meshtype_2/burgers/lc=0.028_ngrid_20_n=5_iso_pad_meshtype_2",
        "./data/dataset_meshtype_6/burgers/lc=0.05_ngrid_20_n=5_iso_pad_meshtype_6",
        "./data/dataset_meshtype_6/burgers/lc=0.028_ngrid_20_n=5_iso_pad_meshtype_6",
    ]

    # ds_roots = [*ds_root_helmholtz, *ds_root_swirl, *ds_root_burgers]
    # ds_roots = [*ds_root_burgers]
    # ds_roots = ['./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6']
    # ds_roots = [*ds_root_swirl, *ds_root_helmholtz]

    ds_roots = [*ds_root_swirl]
    # ds_roots = [*ds_root_helmholtz]
    # ds_roots = [*ds_root_helmholtz]

    # run_ids = [*run_ids_largeset, *run_ids_miniset_new, *run_ids_old_benchmark]
    # run_ids = [*run_ids_miniset_new]
    # run_ids = [run_id_m2n, run_id_m2n_area_loss_hessian_norm, run_id_m2t]
    # run_ids = [run_id_m2t]
    # run_ids = ["cyzk2mna", "u4uxcz1e", "99zrohiu", "gywsmly9"]
    run_ids = ["u4uxcz1e", "99zrohiu", "gywsmly9"]
    # run_ids = ["cyzk2mna", "u4uxcz1e", "gywsmly9"]
    # run_ids = ["99zrohiu", "gywsmly9"]
    # run_ids = ["cyzk2mna"]
    # run_ids = ["u4uxcz1e", "99zrohiu", "gywsmly9"]
    # run_ids = ["ig1np6kx"]
    # run_ids = ["99zrohiu", "ig1np6kx", "u4uxcz1e"]
    # run_ids = ["99zrohiu"]

    # M2N monitor only: 4u40se08
    # MRT monitor only: j821c42y
    # MRT monitor (hessian norm) only: 5r7dfx8n
    # M2T monitor (hessian norm) only: qfi081fd

    run_ids = ["j821c42y", "4u40se08"]
    run_ids = ["yi9cz24h"]  # yi9cz24h, M2T

    run_ids = ["qfi081fd"]
    # run_ids = ["7f9ac3iy"]
    run_ids = ["5r7dfx8n"]
    run_ids = ["yi9cz24h"]
    run_ids = ["j8s7l3kw"]
    run_ids = ["32gs384i"]

    run_ids = ["d9h5uzcp"]
    run_ids = ["yx0h8mfm"]

    run_ids = ["6qa5jius"]  # Ablation of coord trained

    # comparisons
    # M2N large set, small scale, type 6, g86hj04w
    # M2N-en large set, small scale, type 6, c0z773gi
    # M2N-en large set, small scale, type 6, monitor, ta0c8b3u
    # M2N-en large set, small scale, type 2, monitor, 00yvtkg0
    # M2N-en large set, small scale, type 6, area only 7fwne0ig
    # MRT large set, small scale, type 6, yx0h8mfm
    # MRT large set, small scale, type 6, monitor, d9h5uzcp
    # MRT large set, small scale, type 6 coord, 3sicl8ny, deform + area

    # MRT large set, small scale, type 6 coord area only, b2la0gey

    # current best: vnv1mv48

    # M2T large set, small npouut8z
    # run_ids = ["g86hj04w", "yx0h8mfm", "d9h5uzcp"]
    run_ids = ["d9h5uzcp", "yx0h8mfm", "ta0c8b3u", "npouut8z"]

    epoch = 999
    # loginto wandb API
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project_name}")
    latest_run = runs[0]
    print(f"Latest run id {latest_run.id}")
    # run_ids = [latest_run.id]
    # run_ids = ["g86hj04w", "n4t1fqq2"]
    run_ids = ["g86hj04w"]
    run_ids = ["8os8zve7"]  # abalation for solution vs monitor
    run_ids = ["6qa5jius"]  # Ablation of coord trained
    run_ids = ["n4t1fqq2", "6qa5jius", "8os8zve7"]

    # run_ids = ["n4t1fqq2", "g86hj04w"]
    run_ids = ["8os8zve7", "6qa5jius"]
    # run_ids = ["ta0c8b3u"]
    # run_ids = ["7fwne0ig"] # M2N not global
    # chamfer, chamfer 05, chamfer 01
    # run_ids = ["2wjlu0vg", "mk8aaxvz", "m7uar229"]
    # run_ids = ["d2435i9b"]
    # run_ids = ["3sicl8ny"]
    # run_ids = ["g86hj04w"]
    # run_ids = ["d9h5uzcp"]
    # run_ids = ["g86hj04w"]
    for run_id in run_ids:
        for ds_root in ds_roots:
            problem_type, domain, meshtype = get_problem_type(ds_root=ds_root)
            print(f"Evaluating {run_id} on dataset: {ds_root}")

            run = api.run(f"{entity}/{project_name}/{run_id}")
            config = SimpleNamespace(**run.config)

            # Append the monitor val at the end
            # config.mesh_feat.append("monitor_val")
            # config.mesh_feat = ["coord", "u", "monitor_val"]
            # config.mesh_feat = ["coord", "monitor_val"]
            # config.mesh_feat = ["coord", "u"]

            print("# Evaluation Pipeline Started\n")
            print(config)
            # init
            eval_dir = init_dir(config, run_id, epoch, ds_root, problem_type, domain)  # noqa
            dataset = load_dataset(config, ds_root, tar_folder="data")
            model = load_model(config, epoch, eval_dir)

            # bench_res = benchmark_model(
            #     model, dataset, eval_dir, ds_root, start_idx=300, num_samples=100)
            bench_res = benchmark_model(
                model, dataset, eval_dir, ds_root, start_idx=0, num_samples=100
            )

            write_sumo(eval_dir, ds_root)

    exit()

    # ds_roots = ['./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=100_aniso_full_meshtype_6']

    # run_ids = [run_id]
    # run_ids = ['0l8ujpdr', 'hmgwx4ju']
    # run_ids = ['0l8ujpdr']
    # run_ids = ['knjfc14i']
    # run_ids = [run_id]
    # ds_roots = ['./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6']
    # ds_roots = ['./data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<15x15>_n=100_aniso_full',
    #             './data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<20x20>_n=100_aniso_full',
    #             './data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<35x35>_n=100_aniso_full',
    #             './data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2',
    #             './data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6']
    # ds_roots = ['./data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<15x15>_n=100_aniso_full',
    #             './data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6']
    # ds_roots = ['./data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_6']
    # ds_roots = ['./data/dataset_meshtype_0/swirl/z=<0,1>_ndist=None_max_dist=6_<30x30>_n=iso_pad']
    # ds_roots = ['./data/dataset_meshtype_2/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5_meshtype_2']

    # for run_id in run_ids:
    #     for ds_root in ds_roots:
    #         problem_type, domain, meshtype = get_problem_type(ds_root=ds_root)
    #         print(f"Evaluating {run_id} on dataset: {ds_root}")
    #         # loginto wandb API
    #         api = wandb.Api()
    #         run = api.run(f"{entity}/{project_name}/{run_id}")
    #         config = SimpleNamespace(**run.config)

    #         print("# Evaluation Pipeline Started\n")
    #         # init
    #         eval_dir = init_dir(config, run_id, epoch, ds_root)
    #         dataset = load_dataset(config, ds_root, tar_folder='data')
    #         model = load_model(config, epoch, eval_dir)

    #         bench_res = benchmark_model(model, dataset, eval_dir, ds_root)

    #         write_sumo(eval_dir, ds_root)

    # run_id = "5y50gqla"  # M2N og

    # # run_id = "0z6s9vky"  # M2N, with area loss, https://wandb.ai/w-chunyang/warpmesh/runs/xkmmgmrc?workspace=user-w-chunyang  # noqa
    # # run_id = "w7wbgtxa"  # M2N, fine-tuned on hlmltz poly mesh,  https://wandb.ai/mz-team/warpmesh/runs/w7wbgtxa?workspace=user-w-chunyang  # noqa
    # # run_id = "8qhw8kcf"  # M2N, fine-tuned on burgers, https://wandb.ai/mz-team/warpmesh/runs/8qhw8kcf?workspace=user-w-chunyang  # noqa
    # # run_id = ""

    # # run_id = "ayqshvic"  # MRN, with area loss. https://wandb.ai/mz-team/warpmesh/runs/ayqshvic?workspace=user-w-chunyang  # noqa
    # # run_id = "oprm5ns5"  # MRN, fine-tuned on 30 burgers https://wandb.ai/w-chunyang/warpmesh/runs/oprm5ns5?workspace=user-w-chunyang  # noqa
    # # run_id = "gxq23t91"  # MRN, fine-tuned on 30 swirl https://wandb.ai/w-chunyang/warpmesh/runs/gxq23t91?workspace=user-w-chunyang  # noqa
    # # run_id = "hjrebg62"  # MRN, fine-tuned on 30 polymesh https://wandb.ai/w-chunyang/warpmesh/runs/hjrebg62?workspace=user-w-chunyang  # noqa
    # ds_roots = [
    #     # test hard-dataset
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_0/helmholtz/z=<0,1>_ndist=None_max_dist=6_<25x25>_n=100_aniso_full',  # noqa
    #     # test MA reduction
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.06_n=100_aniso_full_meshtype_2'  # noqa

    #     # poisson square
    #     '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/poisson/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/poisson/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/poisson/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_2',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/poisson/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_2',  # noqa

    #     # # poisson poly
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/poisson_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/poisson_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/poisson_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_2',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/poisson_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_2',  # noqa

    #     # helmholtz square
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_2',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_2',  # noqa

    #     # # helmholtz poly
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.055_n=400_aniso_full_meshtype6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_2',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_2/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=400_aniso_full_meshtype_2',  # noqa

    #     # # swirl
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_lc_0.04_interval_5_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_lc_0.045_interval_5_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_lc_0.05_interval_5_meshtype_6',  # noqa

    #     # # burgers
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/burgers/lc=0.04_n=5_iso_pad_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/burgers/lc=0.045_n=5_iso_pad_meshtype_6',  # noqa
    #     # '/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/burgers/lc=0.05_n=5_iso_pad_meshtype_6',  # noqa
    # ]
    # for run_id in run_ids:
    #     for ds_root in ds_roots:
    #         problem_type, domain, meshtype = get_problem_type(ds_root=ds_root)
    #         print(f"Evaluating {run_id} on dataset: {ds_root}")
    #         # loginto wandb API
    #         api = wandb.Api()
    #         run = api.run(f"{entity}/{project_name}/{run_id}")
    #         config = SimpleNamespace(**run.config)

    #         print("# Evaluation Pipeline Started\n")
    #         # init
    #         eval_dir = init_dir(config, run_id, epoch, ds_root, problem_type, domain)  # noqa
    #         dataset = load_dataset(config, ds_root, tar_folder='data')
    #         model = load_model(config, epoch, eval_dir)

    #         # bench_res = benchmark_model(
    #         #     model, dataset, eval_dir, ds_root, start_idx=300, num_samples=100)
    #         bench_res = benchmark_model(
    #             model, dataset, eval_dir, ds_root, start_idx=0, num_samples=100)

    #         write_sumo(eval_dir, ds_root)

    # exit()
