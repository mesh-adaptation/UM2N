# Author: Chunyang Wang
# GitHub Username: chunyang-w

# import packages
import datetime
import glob
import os
from types import SimpleNamespace

import firedrake as fd
import pandas as pd
import torch
import wandb

import UM2N

os.environ["OMP_NUM_THREADS"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

entity = "mz-team"
project_name = "warpmesh"
# run_id = '7py7k3ah' # fine tune on helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2  # noqa
# run_id = 'sr7waaso'  # MRT with no mask
# run_id = 'gl1zpjc5'  # MRN 3-loop

run_id = "8ndi2teh"  # MRN 3-loop, on polymesh

epoch = 999
ds_root = (
    "./data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_lc_0.045_interval_50"  # noqa
)


def init_dir(config):
    """
    Make dir for evaluation. All evaluation files will be stored
    under the dir created.
    """
    print("\t## Making directories to store files")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(project_dir, "eval")
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = os.path.join(
        eval_dir,
        "swirl" + "_" + now + "_" + config.model_used + "_" + str(epoch) + "_" + run_id,
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
    elif config.model_used == "MRT":
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
    else:
        print("Model not found")
    model_file_path = os.path.join(experiment_dir, target_file_name)
    model = UM2N.load_model(model, model_file_path)
    return model


def benchmark_model(model, dataset, eval_dir, ds_root):
    # Readin params for a specific swirl problem
    info_df = pd.read_csv(os.path.join(ds_root, "info.csv"))
    sigma = info_df["sigma"][0]
    alpha = info_df["alpha"][0]
    r_0 = info_df["r_0"][0]
    T = info_df["T"][0]
    n_step = info_df["n_step"][0]

    mesh = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
    mesh_new = fd.Mesh(os.path.join(ds_root, "mesh", "mesh.msh"))
    mesh_fine = fd.Mesh(os.path.join(ds_root, "mesh_fine", "mesh.msh"))

    # fd.triplot(mesh)
    # fd.triplot(mesh_fine)

    evaluator = UM2N.SwirlEvaluator(
        mesh,
        mesh_fine,
        mesh_new,
        dataset,
        model,
        eval_dir,
        ds_root,
        sigma=sigma,
        alpha=alpha,
        r_0=r_0,
        T=T,
        n_step=n_step,
    )

    evaluator.make_log_dir()
    evaluator.make_plot_dir()
    evaluator.make_plot_more_dir()

    eval_res = evaluator.eval_problem()  # noqa

    return


def write_sumo(eval_dir):
    log_dir = os.path.join(eval_dir, "log")
    file_path = os.path.join(log_dir, "log*.csv")
    log_files = glob.glob(file_path)

    error_MA = 0
    error_model = 0
    error_og = 0
    time_MA = 0
    num_tangle = 0
    time_model = 0
    pass_count = 0
    fail_count = 0
    total_count = 0

    for file_names in log_files:
        total_count += 1
        log_df = pd.read_csv(file_names)
        if log_df["tangled_element"][0] == 0:
            pass_count += 1
            error_og += log_df["error_og"][0]
            error_MA += log_df["error_ma"][0]
            error_model += log_df["error_model"][0]
            time_MA += log_df["time_consumption_MA"][0]
            time_model += log_df["time_consumption_model"][0]
        else:
            fail_count += 1
            num_tangle += log_df["tangled_element"][0]

    sumo_df = pd.DataFrame(
        {
            "error_reduction_MA": (error_og - error_MA) / error_og,
            "error_reduction_model": (error_og - error_model) / error_og,
            "time_MA": time_MA,
            "time_model": time_model,
            "acceleration_ratio": time_MA / time_model,
            "tangle_total": num_tangle,
            "TEpM(tangled Elements per Mesh)": num_tangle / total_count,
            "failed_case": fail_count,
            "total_case": total_count,
            "dataset_path": ds_root,
        },
        index=[0],
    )
    sumo_df.to_csv(os.path.join(eval_dir, "sumo.csv"))


if __name__ == "__main__":
    print("Evaluation Script for 2D Ring Problem \n")
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    eval_dir = init_dir(config)
    dataset = load_dataset(config, ds_root, tar_folder="data")
    model = load_model(config, epoch, eval_dir)

    bench_res = benchmark_model(model, dataset, eval_dir, ds_root)

    write_sumo(eval_dir)

    exit()
