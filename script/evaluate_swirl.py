# Author: Chunyang Wang
# GitHub Username: chunyang-w

# import packages
import datetime
import glob
import time
import torch
import os
import wandb
import pprint

import firedrake as fd
import matplotlib.pyplot as plt
import pandas as pd
import warpmesh as wm

from torch_geometric.loader import DataLoader
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

entity = 'w-chunyang'
# entity = 'mz-team'
project_name = 'warpmesh'
# run_id = '7py7k3ah' # fine tune on helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2
# run_id = 'sr7waaso'  # MRT with no mask
# run_id = 'gl1zpjc5'  # MRN 3-loop
# run_id = '3wv8mgyt'  # MRN 3-loop, on polymesh
run_id = '55tlmka8'  # MRN 3-loop on type6 mesh
epoch = 399

ds_root = "/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/swirl/sigma_0.017_alpha_1.5_r0_0.2_lc_0.05_interval_5_meshtype_6/"  # noqa


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

    # fd.triplot(mesh)
    # fd.triplot(mesh_fine)

    evaluator = wm.SwirlEvaluator(
        mesh, mesh_fine, mesh_new, dataset, model, eval_dir, ds_root,
        sigma=sigma, alpha=alpha, r_0=r_0,
        T=T, n_step=n_step,
    )

    evaluator.make_log_dir()
    evaluator.make_plot_dir()
    evaluator.make_plot_more_dir()

    eval_res = evaluator.eval_problem()                     # noqa

    return


def write_sumo(eval_dir):
    pass


dummy_config_raw = {
    'batch_size': 10,
    'check_tangle_interval': 10,
    'conv_feat': ['conv_uh', 'conv_hessian_norm'],
    'conv_feat_fix': ['conv_uh_fix'],
    'count_tangle_method': 'inversion',
    'experiment_name': '2024-01-11-11:56_MRN_area_loss_bi_edge_poly',
    'inversion_loss_scaler': 10000000000.0,
    'is_normalise': True,
    'lc_test': [0.045, 0.04],
    'lc_train': [0.055, 0.05],
    'learning_rate': 5e-05,
    'mesh_feat': ['coord', 'u', 'hessian_norm'],
    'model_used': 'MRN',
    'multi_scale_check_interval': 50,
    'num_deform_in': 3,
    'num_deformer_loop': 3,
    'num_epochs': 1500,
    'num_gfe_in': 2,
    'num_lfe_in': 4,
    'out_path': '/content/drive/MyDrive/warpmesh/out',
    'print_interval': 1,
    'project': 'warpmesh',
    'save_interval': 20,
    'train_boundary_scheme': 'full',
    'train_data_set_type': 'aniso',
    'use_area_loss': True,
    'use_inversion_diff_loss': False,
    'use_inversion_loss': False,
    'use_jacob': False,
    'weight_decay': 0.1,
    'x_feat': ['coord', 'bd_mask']
}


dummy_config = SimpleNamespace(**dummy_config_raw)


def get_local_model(config):
    model_path = "/Users/chunyang/projects/WarpMesh/temp/model_599.pth"
    model = wm.MRN(
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        deform_in_c=config.num_deform_in,
        num_loop=config.num_deformer_loop,
    )
    model = wm.load_model(model, model_path)
    return model


if __name__ == "__main__":
    print("Evaluation Script for 2D Ring Problem \n")
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    # config = dummy_config
    eval_dir = init_dir(config)
    dataset = load_dataset(config, ds_root, tar_folder='data')
    model = load_model(config, epoch, eval_dir)
    # model = get_local_model(config)

    bench_res = benchmark_model(model, dataset, eval_dir, ds_root)

    write_sumo(eval_dir)

    exit()
