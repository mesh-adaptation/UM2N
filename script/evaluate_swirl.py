# Author: Chunyang Wang
# GitHub Username: chunyang-w

# import packages
import datetime
import glob
import time
import torch
import os
import wandb

import firedrake as fd
import matplotlib.pyplot as plt
import pandas as pd
import warpmesh as wm

from torch_geometric.loader import DataLoader
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

entity = 'w-chunyang'
project_name = 'warpmesh'
# run_id = 'sr7waaso'  # MRT with no mask
# run_id = 'gl1zpjc5'  # MRN 3-loop
run_id = '3wv8mgyt'  # MRN 3-loop, on polymesh

epoch = 599
ds_root = "/Users/chunyang/projects/WarpMesh/data/dataset/swirl/sigma_0.017_alpha_1.5_r0_0.2_lc_0.045_interval_50"  # noqa


def init_dir(config):
    """
    Make dir for evaluation. All evaluation files will be stored
    under the dir created.
    """
    print("\t## Making directories to store files")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    eval_dir = os.path.join(project_dir, 'eval')
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = os.path.join(
        eval_dir, 'swirl' + '_' + now + '_' + config.model_used + '_'
        + str(epoch) + '_' + run_id)
    wm.mkdir_if_not_exist(experiment_dir)
    print("\t## Make eval dir done\n")
    return experiment_dir


def load_dataset(config, ds_root, tar_folder,
                 use_run_time_cluster=False,
                 use_cluster=False):
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
    dataset = wm.MeshDataset(
        dataset_path,
        transform=wm.normalise if wm.normalise else None,
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
    '''
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
    '''

    target_file_name = "model_{}.pth".format(epoch)
    model_file = None
    for file in run.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(root=experiment_dir, replace=True)
    assert model_file is not None, "Model file not found"
    model = None
    if (config.model_used == "M2N"):
        model = wm.M2N(
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
            deform_in_c=config.num_deform_in,
        )
    elif (config.model_used == "MRN"):
        model = wm.MRN(
            gfe_in_c=config.num_gfe_in,
            lfe_in_c=config.num_lfe_in,
            deform_in_c=config.num_deform_in,
            num_loop=config.num_deformer_loop,
        )
    elif (config.model_used == "MRT"):
        model = wm.MRTransformer(
            num_transformer_in=config.num_transformer_in,
            num_transformer_out=config.num_transformer_out,
            num_transformer_embed_dim=config.num_transformer_embed_dim,
            num_transformer_heads=config.num_transformer_heads,
            num_transformer_layers=config.num_transformer_layers,
            transformer_training_mask=config.transformer_training_mask,
            transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,  # noqa
            transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,  # noqa
            deform_in_c=config.num_deform_in,
            num_loop=config.num_deformer_loop,
            device=device,
        )
    else:
        print("Model not found")
    model_file_path = os.path.join(experiment_dir, target_file_name)
    model = wm.load_model(model, model_file_path)
    return model


def benchmark_model(model, dataset, eval_dir, ds_root,
                    start_idx=0, num_samples=100
                    ):
    # Readin params for a specific swirl problem
    info_df = pd.read_csv(os.path.join(ds_root, 'info.csv'))
    sigma = info_df["sigma"][0]
    alpha = info_df["alpha"][0]
    r_0 = info_df["r_0"][0]
    T = info_df["T"][0]
    n_step = info_df["n_step"][0]

    mesh = fd.Mesh(os.path.join(ds_root, 'mesh', 'mesh.msh'))
    mesh_new = fd.Mesh(os.path.join(ds_root, 'mesh', 'mesh.msh'))
    mesh_fine = fd.Mesh(os.path.join(ds_root, 'mesh_fine', 'mesh.msh'))

    evaluator = wm.SwirlEvaluator(
        mesh, mesh_fine, mesh_new, dataset,
        sigma=sigma, alpha=alpha, r_0=r_0,
        T=T, n_step=n_step,
    )
    evaluator.eval_problem()

    return


def write_sumo(eval_dir):
    pass


if __name__ == "__main__":
    print("Evaluation Script for 2D Ring Problem \n")
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    eval_dir = init_dir(config)
    dataset = load_dataset(config, ds_root, tar_folder='data')
    model = load_model(config, epoch, eval_dir)

    bench_res = benchmark_model(model, dataset, eval_dir, ds_root)
    write_sumo(eval_dir)
    exit()
