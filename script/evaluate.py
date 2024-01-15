# Author: Chunyang Wang
# GitHub Username: acse-cw1722

# A evaluation pipeline script for evaluating the performance of input model:
#
#       1. Calculate PDE loss
#       2. Evaluate Deform Loss
#       3. Monitor time consumed

# import packages
import datetime
import glob
import time
import torch
import os
import wandb

import firedrake as fd
import matplotlib.pyplot as plt # noqa
import pandas as pd
import warpmesh as wm

from torch_geometric.loader import DataLoader
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

entity = 'w-chunyang'
project_name = 'warpmesh'
# run_id = 'sr7waaso'  # MRT with no mask
# run_id = 'gl1zpjc5'  # MRN 3-loop
run_id = '3wv8mgyt'  # MRN 3-loop, on polymesh

epoch = 599
ds_root = (  # square
        '/Users/chunyang/projects/WarpMesh/data/dataset/helmholtz/'
        'z=<0,1>_ndist=None_max_dist=6_<25x25>_n=100_aniso_full')
# ds_root = (  # poly
#     '/Users/chunyang/projects/WarpMesh/data/dataset/helmholtz_poly'
#     '/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=400_aniso_full'
# )
# ds_root = (  # poly
#     '/Users/chunyang/projects/WarpMesh/data/dataset/poisson_poly'
#     '/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=400_aniso_full'
# )


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
        eval_dir, now + '_' + config.model_used + '_'
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


def get_log_og(log_path, idx):
    """
    Read log file from dataset log dir and return value in it
    """
    df = pd.read_csv(os.path.join(log_path, f"log{idx}.csv"))
    return {
        "error_og": df["error_og"][0],
        "error_adapt": df["error_adapt"][0],
        "time": df["time"][0],
    }


def benchmark_model(model, dataset, eval_dir, ds_root,
                    start_idx=0, num_samples=100
                    ):
    """"
    benchmark the performance of a model on a specific data
    set.

    Args:
        data_path (str): the path to the datasets root to evaluate.
        tar_folder (str): the target folder name containing the input samples.
        eval_dir (str): path to the evalution dir a.k.a. data root.
        num_samples (int): number of samples used to do the evaluation.
    """
    domain = None
    ds_type = ds_root.split('/')[-2]
    problem_list = ds_type.split('_')
    problem_type = problem_list[0]
    if (len(problem_list) == 2):
        if (problem_list[-1] == 'poly'):
            domain = 'poly'
    else:
        domain = 'square'
    print('problem type: ', problem_type, domain)

    ds_info_df_path = os.path.join(ds_root, 'info.csv')
    df_info_df = pd.read_csv(ds_info_df_path)
    n_grid = None
    mesh = None
    mesh_fine = None
    if domain == 'square':
        n_grid = df_info_df['n_grid'][0]

    log_dir = os.path.join(eval_dir, 'log')
    plot_dir = os.path.join(eval_dir, 'plot')
    wm.mkdir_if_not_exist(log_dir)
    wm.mkdir_if_not_exist(plot_dir)

    for idx in range(start_idx, start_idx + num_samples):
        # model inference stage
        sample = next(iter(
            DataLoader([dataset[idx]], batch_size=1, shuffle=False)))
        model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            out = model(sample, poly_mesh=True if domain == "poly" else False)
            end = time.perf_counter()
            dur_ms = (end - start) * 1000
        temp_time_consumption = dur_ms
        temp_tangled_elem = wm.get_sample_tangle(out, sample.y, sample.face)
        temp_loss = 1000 * torch.nn.L1Loss()(out, sample.y)
        # define mesh & fine mesh for comparison
        if domain == 'square':
            mesh = fd.UnitSquareMesh(n_grid, n_grid)
            mesh_MA = fd.UnitSquareMesh(n_grid, n_grid)
            mesh_fine = fd.UnitSquareMesh(80, 80)
            mesh_model = fd.UnitSquareMesh(n_grid, n_grid)
        elif domain == 'poly':
            mesh = fd.Mesh(
                os.path.join(ds_root, 'mesh', f'mesh{idx}.msh'))
            mesh_MA = fd.Mesh(
                os.path.join(ds_root, 'mesh', f'mesh{idx}.msh'))
            mesh_fine = fd.Mesh(
                os.path.join(ds_root, 'mesh_fine', f'mesh{idx}.msh'))
            mesh_model = fd.Mesh(
                os.path.join(ds_root, 'mesh', f'mesh{idx}.msh'))
        mesh_model.coordinates.dat.data[:] = out.detach().cpu().numpy()

        compare_res = wm.compare_error(
            sample, mesh, mesh_fine, mesh_model, mesh_MA, temp_tangled_elem,
            problem_type=problem_type)
        temp_error_model = compare_res["error_model_mesh"]
        temp_error_og = compare_res["error_og_mesh"]
        temp_error_ma = compare_res["error_ma_mesh"]

        log_og = get_log_og(os.path.join(ds_root, 'log'), idx)
        log_error_og = log_og["error_og"]
        log_error_ma = log_og["error_adapt"]
        log_time = log_og["time"]

        # log file
        log_df = pd.DataFrame({
            'deform_loss': temp_loss.numpy(),
            'tangled_element': temp_tangled_elem,
            'error_og': temp_error_og,
            'error_model': temp_error_model,
            'error_ma': temp_error_ma,
            'error_log_og': log_error_og,
            'error_log_ma': log_error_ma,
            'error_reduction_MA': (temp_error_og - temp_error_ma) / temp_error_og, # noqa
            'error_reduction_model': (temp_error_og - temp_error_model) / temp_error_og, # noqa
            'time_consumption_model': temp_time_consumption,
            'time_consumption_MA': log_time,
            'acceration_ratio': log_time / temp_time_consumption,
        }, index=[0])
        log_df.to_csv(os.path.join(log_dir, f"log{idx}.csv"))

        fig = wm.plot_mesh_compare_benchmark(
            out, sample.y, sample.face,
            temp_loss, temp_tangled_elem
        )
        fig.savefig(
            os.path.join(plot_dir, f"plot_{idx}.png")
        )


def write_sumo(eval_dir):
    log_dir = os.path.join(eval_dir, 'log')
    file_path = os.path.join(log_dir, 'log*.csv')
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
        if log_df['tangled_element'][0] == 0:
            pass_count += 1
            error_og += log_df['error_og'][0]
            error_MA += log_df['error_ma'][0]
            error_model += log_df['error_model'][0]
            time_MA += log_df['time_consumption_MA'][0]
            time_model += log_df['time_consumption_model'][0]
        else:
            fail_count += 1
            num_tangle += log_df['tangled_element'][0]

    sumo_df = pd.DataFrame({
        'error_reduction_MA': (error_og - error_MA) / error_og,
        'error_reduction_model': (error_og - error_model) / error_og,
        'time_MA': time_MA,
        'time_model': time_model,
        'acceleration_ratio': time_MA / time_model,
        'tangle_total': num_tangle,
        'TEpM(tangled Elements per Mesh)': num_tangle / total_count,
        'failed_case': fail_count,
        'total_case': total_count,
        'dataset_path': ds_root,
    }, index=[0])
    sumo_df.to_csv(os.path.join(eval_dir, 'sumo.csv'))


if __name__ == "__main__":
    # loginto wandb API
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)

    print("# Evaluation Pipeline Started\n")
    # init
    eval_dir = init_dir(config)
    dataset = load_dataset(config, ds_root, tar_folder='data')
    model = load_model(config, epoch, eval_dir)

    bench_res = benchmark_model(model, dataset, eval_dir, ds_root)
    write_sumo(eval_dir)

    exit()
