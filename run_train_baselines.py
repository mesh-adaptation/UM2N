# package import
# %load_ext autoreload
# %autoreload 2

# from google.colab import userdata
import argparse

# from google.colab import runtime
import os
import warnings
from datetime import datetime

import numpy as np
import torch
import wandb
from torch_geometric.data import DataLoader

from UM2N.helper import load_yaml_to_namespace, save_namespace_to_yaml
from UM2N.loader import AggreateDataset, MeshDataset, normalise
from UM2N.model import (
    M2N,
    M2N_T,
    M2T,
    MRN,
    MRTransformer,
    evaluate,
    evaluate_unsupervised,
    train,
    train_unsupervised,
)

random_seed = 666

torch.manual_seed(random_seed)
np.random.seed(random_seed)


parser = argparse.ArgumentParser(
    prog="UM2N", description="warp the mesh", epilog="warp the mesh"
)
parser.add_argument("-config", default="", type=str, required=True)
args = parser.parse_args()


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.login(key=userdata.get("wandb_key"))
wandb.login(key="9e49ed1812a0349724515be9c3c856f4b1c86cad")

config_name = args.config
config = load_yaml_to_namespace(f"./configs/{config_name}")

# Define path where data get stored
now = datetime.now()
now_date = now.strftime("%Y-%m-%d-%H:%M_")
config.experiment_name = now_date + config_name

model = None
if config.model_used == "MRTransformer":
    model = MRTransformer(
        num_transformer_in=config.num_transformer_in,
        num_transformer_out=config.num_transformer_out,
        num_transformer_embed_dim=config.num_transformer_embed_dim,
        num_transformer_heads=config.num_transformer_heads,
        num_transformer_layers=config.num_transformer_layers,
        transformer_training_mask=config.transformer_training_mask,
        transformer_key_padding_training_mask=config.transformer_key_padding_training_mask,
        transformer_attention_training_mask=config.transformer_attention_training_mask,
        transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,
        transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,
        deform_in_c=config.num_deform_in,
        deform_out_type=config.deform_out_type,
        num_loop=config.num_deformer_loop,
        device=device,
    )
elif config.model_used == "M2T":
    model = M2T(
        num_transformer_in=config.num_transformer_in,
        num_transformer_out=config.num_transformer_out,
        num_transformer_embed_dim=config.num_transformer_embed_dim,
        num_transformer_heads=config.num_transformer_heads,
        num_transformer_layers=config.num_transformer_layers,
        transformer_training_mask=config.transformer_training_mask,
        transformer_key_padding_training_mask=config.transformer_key_padding_training_mask,
        transformer_attention_training_mask=config.transformer_attention_training_mask,
        transformer_training_mask_ratio_lower_bound=config.transformer_training_mask_ratio_lower_bound,
        transformer_training_mask_ratio_upper_bound=config.transformer_training_mask_ratio_upper_bound,
        deform_in_c=config.num_deform_in,
        deform_out_type=config.deform_out_type,
        local_feature_dim_in=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
        device=device,
    )
elif config.model_used == "M2N":
    model = M2N(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
elif config.model_used == "MRN":
    model = MRN(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
elif config.model_used == "M2N_T":
    model = M2N_T(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
else:
    raise Exception(f"Model {config.model_used} not implemented.")

# =================== load from checkpoint ==========================

if hasattr(config, "use_pre_train") and config.use_pre_train:
    import UM2N

    # Load from checkpoint
    entity = "mz-team"
    project_name = "warpmesh"
    # run_id = "rud1gsge"
    # run_id = "kgr0nicn"
    # run_id = "j8s7l3kw"  # MRN train on monitor val only
    run_id = (
        "3sicl8ny"  # MRN train on monitor val only, mesh type 6, 0.05, 0.055, coord
    )
    # run_id = "99zrohiu"
    api = wandb.Api()
    run_loaded = api.run(f"{entity}/{project_name}/{run_id}")
    epoch = 999
    target_file_name = "model_{}.pth".format(epoch)
    model_file = None
    model_store_path = "./fine_tune_model"
    for file in run_loaded.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(root=model_store_path, replace=True)
            target_file_name = file.name
    assert model_file is not None, "Model file not found"
    model_file_path = os.path.join(model_store_path, target_file_name)

    # # TODO: ad hoc solution for removing the mlp in
    # pretrained_dict = torch.load(model_file_path)
    # model_dict = model.state_dict()
    # pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items()
    #     if (k in model_dict and k != "transformer_encoder.mlp_in.layers.0.weight")
    # }
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    model = UM2N.load_model(model, model_file_path, strict=False)
    print(f"Model {run_id} checkpoint loaded.")
else:
    print("No pre-train. Train from scratch.")
# ===================================================================


############### Change This To Dataset folder #################
data_root = config.data_root

# data set for training
# data_paths = [
#     "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6",
#     # "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.055_n=100_aniso_full_meshtype_2",
#     # "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.045_n=300_aniso_full_meshtype_2",
# ]
data_paths = [f"{pth}" for pth in config.data_root]
###############################################################


loss_func = torch.nn.L1Loss()

print(model)
print()
print(data_paths)

# Load datasets
train_sets = [
    MeshDataset(
        os.path.join(data_path, "train"),
        transform=normalise if config.is_normalise else None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        load_jacobian=config.use_jacob,
        use_cluster=config.use_cluster,
        r=config.cluster_r,
    )
    for data_path in data_paths
]

test_sets = [
    MeshDataset(
        os.path.join(data_path, "test"),
        transform=normalise if config.is_normalise else None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        load_jacobian=config.use_jacob,
        use_cluster=config.use_cluster,
        r=config.cluster_r,
    )
    for data_path in data_paths
]

val_sets = [
    MeshDataset(
        os.path.join(data_path, "val"),
        transform=normalise if config.is_normalise else None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        load_jacobian=config.use_jacob,
        use_cluster=config.use_cluster,
        r=config.cluster_r,
    )
    for data_path in data_paths
]

# for training, datasets preperation
train_set = AggreateDataset(train_sets)
test_set = AggreateDataset(test_sets)
# val_set = AggreateDataset(val_sets)

# Loading and Batching
train_loader = DataLoader(train_set, batch_size=config.batch_size)
test_loader = DataLoader(test_set, batch_size=config.batch_size)
# val_loader = DataLoader(val_set, batch_size=batch_size)

# =============================================== Run training ===============================
# start wandb session
run = wandb.init(
    project=config.project,
    name=config.experiment_name,
    tags=[config.model_used],
    config=config.__dict__,
)
# artifact = wandb.Artifact(name=config.experiment_name.replace(':', '_'), type="model")

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=config.weight_decay,
)

# The model, moving to training device
model = model.to(device)

# arrays for wandb plotting

# Construct a folder for storing trained models locally as backups
output_folder = os.path.join(config.out_path, config.experiment_name)
os.makedirs(output_folder, exist_ok=True)

save_namespace_to_yaml(config, f"{output_folder}/{config.experiment_name}")


for epoch in range(config.num_epochs + 1):
    if config.model_used == "MRTransformer" or config.model_used == "M2T":
        weight_chamfer_loss = 0.0
        if "weight_chamfer_loss" in config:
            weight_chamfer_loss = config.weight_chamfer_loss
        train_func = train_unsupervised
        evaluate_func = evaluate_unsupervised
        train_loss = train_func(
            train_loader,
            model,
            optimizer,
            device,
            loss_func=loss_func,
            use_area_loss=config.use_area_loss,
            use_convex_loss=config.use_convex_loss,
            use_add_random_query=config.use_add_random_query,
            # use_equation_residual_on_grid=config.use_equation_residual_on_grid,
            finite_difference_grad=config.finite_difference_grad,
            weight_area_loss=config.weight_area_loss,
            weight_deform_loss=config.weight_deform_loss,
            weight_eq_residual_loss=config.weight_eq_residual_loss,
            weight_chamfer_loss=weight_chamfer_loss,
            scaler=300,
        )
        test_loss = evaluate_func(
            test_loader,
            model,
            device,
            loss_func=loss_func,
            use_area_loss=config.use_area_loss,
            use_convex_loss=config.use_convex_loss,
            use_add_random_query=config.use_add_random_query,
            # use_equation_residual_on_grid=config.use_equation_residual_on_grid,
            finite_difference_grad=config.finite_difference_grad,
            weight_area_loss=config.weight_area_loss,
            weight_deform_loss=config.weight_deform_loss,
            weight_eq_residual_loss=config.weight_eq_residual_loss,
            weight_chamfer_loss=weight_chamfer_loss,
            scaler=300,
        )
    elif config.model_used == "M2N" or config.model_used == "M2N_T":
        weight_area_loss = 1.0
        weight_deform_loss = 1.0
        if "weight_deform_loss" in config:
            weight_deform_loss = config.weight_deform_loss
        if "weight_area_loss" in config:
            weight_area_loss = config.weight_area_loss

        weight_chamfer_loss = 0.0
        if "weight_chamfer_loss" in config:
            weight_chamfer_loss = config.weight_chamfer_loss

        train_func = train
        evaluate_func = evaluate
        train_loss = train_func(
            train_loader,
            model,
            optimizer,
            device,
            loss_func=loss_func,
            use_area_loss=config.use_area_loss,
            weight_deform_loss=weight_deform_loss,
            weight_area_loss=weight_area_loss,
            weight_chamfer_loss=weight_chamfer_loss,
            scaler=300,
        )
        test_loss = evaluate_func(
            test_loader,
            model,
            device,
            loss_func=loss_func,
            use_area_loss=config.use_area_loss,
            weight_deform_loss=weight_deform_loss,
            weight_area_loss=weight_area_loss,
            weight_chamfer_loss=weight_chamfer_loss,
            scaler=300,
        )
    elif config.model_used == "MRN":
        train_func = train
        evaluate_func = evaluate
        train_loss = train_func(
            train_loader,
            model,
            optimizer,
            device,
            loss_func=loss_func,
            use_area_loss=config.use_area_loss,
            scaler=300,
        )
        test_loss = evaluate_func(
            test_loader,
            model,
            device,
            loss_func=loss_func,
            use_area_loss=config.use_area_loss,
            scaler=300,
        )
    wandb.log(
        {
            "Deform Loss/Train": train_loss["deform_loss"],
            "Deform Loss/Test": test_loss["deform_loss"],
        },
        step=epoch,
    )

    if "convex_loss" in train_loss:
        train_convex_loss = train_loss["convex_loss"]
    else:
        train_convex_loss = 0.0
    if "convex_loss" in test_loss:
        test_convex_loss = test_loss["convex_loss"]
    else:
        test_convex_loss = 0.0

    wandb.log(
        {
            "Convex loss/Train": train_convex_loss,
            "Convex loss/Test": test_convex_loss,
        },
        step=epoch,
    )

    if config.use_inversion_loss:
        wandb.log(
            {
                "Inversion Loss/Train": train_loss["inversion_loss"],
                "Inversion Loss/Test": test_loss["inversion_loss"],
            },
            step=epoch,
        )
    if config.use_area_loss:
        wandb.log(
            {
                "Area Loss/Train": train_loss["area_loss"],
                "Area Loss/Test": test_loss["area_loss"],
            },
            step=epoch,
        )
    train_deform_loss = train_loss["deform_loss"]
    test_deform_loss = test_loss["deform_loss"]

    if "chamfer_loss" in train_loss:
        train_chamfer_loss = train_loss["chamfer_loss"]
    else:
        train_chamfer_loss = 0.0
    if "chamfer_loss" in test_loss:
        test_chamfer_loss = test_loss["chamfer_loss"]
    else:
        test_chamfer_loss = 0.0

    wandb.log(
        {
            "Chamfer loss/Train": train_chamfer_loss,
            "Chamfer loss/Test": test_chamfer_loss,
        },
        step=epoch,
    )

    if "equation_residual" in train_loss:
        train_equation_residual = train_loss["equation_residual"]
    else:
        train_equation_residual = 0.0
    if "equation_residual" in test_loss:
        test_equation_residual = test_loss["equation_residual"]
    else:
        test_equation_residual = 0.0

    wandb.log(
        {
            "Equation residual/Train": train_equation_residual,
            "Equation residual/Test": test_equation_residual,
        },
        step=epoch,
    )

    if config.use_area_loss:
        train_area_loss = train_loss["area_loss"]
        test_area_loss = test_loss["area_loss"]
        print(
            f"Epoch: {epoch} Train deform loss: {train_deform_loss:.4f} Train chamfer loss: {train_chamfer_loss:.4f} Train area loss: {train_area_loss:.4f} Test deform loss: {test_deform_loss:.4f} Test chamfer loss: {test_chamfer_loss:.4f} Test area loss: {test_area_loss:.4f}"
        )
    else:
        print(
            f"Epoch: {epoch} Train deform loss: {train_deform_loss:.4f} Train chamfer loss: {train_chamfer_loss:.4f} Test deform loss: {test_deform_loss:.4f} Test chamfer loss: {test_chamfer_loss:.4f}"
        )

    #   if (epoch) % config.check_tangle_interval == 0:
    #       train_tangle = count_dataset_tangle(train_set, model, device, method=config.count_tangle_method)
    #       test_tangle = count_dataset_tangle(test_set, model, device, method=config.count_tangle_method)
    #       wandb.log({
    #           "Tangled Elements per Mesh/Train": train_tangle,
    #           "Tangled Elements per Mesh/Test":test_tangle,
    #       }, step=epoch)

    if (epoch + 1) % config.save_interval == 0:
        torch.save(model.state_dict(), "{}/model_{}.pth".format(output_folder, epoch))
        # artifact.add_file(local_path="model_{}.pth".format(epoch))
        wandb.save("{}/model_{}.pth".format(output_folder, epoch))
# run.log_artifact(artifact)
run.finish()
