# Author: Chunyang Wang
# GitHub Username: chunyang-w

# import packages
import datetime
import glob
import torch
import os
import wandb

import firedrake as fd
import pandas as pd
import warpmesh as wm

from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

entity = 'w-chunyang'
# entity = 'mz-team'
project_name = 'warpmesh'
# run_id = '7py7k3ah' # fine tune on helmholtz z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2  # noqa
# run_id = 'sr7waaso'  # MRT with no mask
# run_id = 'gl1zpjc5'  # MRN 3-loop
# run_id = '3wv8mgyt'  # MRN 3-loop, on polymesh
run_id = '55tlmka8'  # MRN 3-loop on type6 mesh
epoch = 1499

ds_root = "/Users/chunyang/projects/WarpMesh/data/dataset_meshtype_6/burgers/lc=0.05_n=5_iso_pad_meshtype_6/"  # noqa


def get_sample_param_of_nu_generalization_by_idx_train(idx_in):
    gauss_list_ = []
    if idx_in == 1:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 2:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 3:
        param_ = {
            "cx": 0.225,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 4:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 5:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 6:
        shift_ = 0.15
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    elif idx_in == 7:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.0001
    elif idx_in == 8:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.001
    elif idx_in == 9:
        shift_ = 0.2
        param_ = {
            "cx": 0.3,
            "cy": 0.5 + shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.3,
            "cy": 0.5 - shift_,
            "w": 0.01}
        gauss_list_.append(param_)
        param_ = {
            "cx": 0.15,
            "cy": 0.5,
            "w": 0.01}
        gauss_list_.append(param_)
        nu_ = 0.002
    return gauss_list_, nu_


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
        eval_dir, 'burgers' + '_' + now + '_' + config.model_used + '_'
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


def benchmark_model(model, dataset, eval_dir, ds_root, case_idxs):
    # Select params to generate burgers bump
    for idx in case_idxs:
        gaussian_list, nu = get_sample_param_of_nu_generalization_by_idx_train(idx)  # noqa
        mesh = fd.Mesh(os.path.join(ds_root, 'mesh', 'mesh.msh'))
        mesh_new = fd.Mesh(os.path.join(ds_root, 'mesh', 'mesh.msh'))
        mesh_fine = fd.Mesh(os.path.join(ds_root, 'mesh_fine', 'mesh.msh'))

        # fd.triplot(mesh)
        # fd.triplot(mesh_fine)

        evaluator = wm.BurgersEvaluator(
            mesh, mesh_fine, mesh_new,
            dataset, model, eval_dir, ds_root, idx,
            gauss_list=gaussian_list, nu=nu
        )

        evaluator.make_log_dir()
        evaluator.make_plot_dir()
        evaluator.make_plot_more_dir()

        eval_res = evaluator.eval_problem()                     # noqa

        return


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
    print("Evaluation Script for Burgers PDE \n")
    # a list containing a series of number between 1-9
    # controling which cases are to be evaluated.
    # usually between 1-5
    # (the maximum number should be 'n_case' set in 'build_burgers_square.py')

    case_idxs = [2]
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    eval_dir = init_dir(config)
    dataset = load_dataset(config, ds_root, tar_folder='data')
    model = load_model(config, epoch, eval_dir)

    bench_res = benchmark_model(model, dataset, eval_dir, ds_root, case_idxs)

    write_sumo(eval_dir)

    exit()
