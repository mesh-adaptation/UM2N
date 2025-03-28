# package import
# %load_ext autoreload
# %autoreload 2

# from google.colab import userdata
import argparse
import gc

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
    MRN,
    M2N_dynamic_drop,
    M2N_dynamic_no_drop,
    M2NAtten,
    M2Transformer,
    MRNAtten,
    MRNGlobalTransformerEncoder,
    MRNLocalTransformerEncoder,
    MRTransformer,
    count_dataset_tangle,
    evaluate_unsupervised,
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

# config_name = "MRN_area_loss_bi_edge"
# config_name = "MRN_GTE_area_loss_bi_edge"
# config_name = "MRT_area_loss_bi_edge"
config_name = args.config
config = load_yaml_to_namespace(f"./configs/{config_name}")

# Define path where data get stored
now = datetime.now()
now_date = now.strftime("%Y-%m-%d-%H:%M_")
config.experiment_name = now_date + config_name

model = None
if config.model_used == "M2N":
    model = M2N(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
elif config.model_used == "M2NAtten":
    model = M2NAtten(
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
elif config.model_used == "M2N_dynamic_drop":
    model = M2N_dynamic_drop(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
elif config.model_used == "M2N_dynamic_no_drop":
    model = M2N_dynamic_no_drop(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
elif config.model_used == "MRNAtten":
    model = MRNAtten(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
elif config.model_used == "MRNGlobalTransformerEncoder":
    model = MRNGlobalTransformerEncoder(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
elif config.model_used == "MRNLocalTransformerEncoder":
    model = MRNLocalTransformerEncoder(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
elif config.model_used == "MRTransformer":
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
elif config.model_used == "M2Transformer":
    model = M2Transformer(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
else:
    raise Exception(f"Model {config.model_used} not implemented.")


############### Change This To Dataset folder #################
data_root = config.data_root

# data set for training
data_paths = [
    (
        f"{data_root}"
        f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=400_"
        f"{config.train_data_set_type}_"
        f"{config.train_boundary_scheme}"
    )
    for n_grid in config.n_grids
]

# data set for testing on different mesh scales
data_paths_iso_pad = [
    (f"{data_root}" f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=100_iso_pad")
    for n_grid in config.n_grids_test
]
data_paths_iso_full = [
    (f"{data_root}" f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=100_iso_full")
    for n_grid in config.n_grids_test
]
data_paths_aniso_pad = [
    (
        f"{data_root}"
        f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=100_aniso_pad"
    )
    for n_grid in config.n_grids_test
]
data_paths_aniso_full = [
    (
        f"{data_root}"
        f"z=<0,1>_ndist=None_max_dist=6_<{n_grid}x{n_grid}>_n=100_aniso_full"
    )
    for n_grid in config.n_grids_test
]
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

# for testing on multiple mesh scale, datasets and batch loader:
# iso_pad_sets = [MeshDataset(
#     os.path.join(data_path, "data"),
#     transform=normalise if config.is_normalise else None,
#     x_feature=config.x_feat,
#     mesh_feature=config.mesh_feat,
#     conv_feature=config.conv_feat,
#     conv_feature_fix=config.conv_feat_fix,
#     load_jacobian=config.use_jacob,
#     use_cluster=config.use_cluster,
#     r=config.cluster_r,
# ) for data_path in data_paths_iso_pad]

# iso_pad_loaders = [DataLoader(test_set_i, batch_size=config.batch_size) for test_set_i in iso_pad_sets]

# iso_full_sets = [MeshDataset(
#     os.path.join(data_path, "data"),
#     transform=normalise if config.is_normalise else None,
#     x_feature=config.x_feat,
#     mesh_feature=config.mesh_feat,
#     conv_feature=config.conv_feat,
#     conv_feature_fix=config.conv_feat_fix,
#     load_jacobian=config.use_jacob,
#     use_cluster=config.use_cluster,
#     r=config.cluster_r,
# ) for data_path in data_paths_iso_full]

# iso_full_loaders = [DataLoader(test_set_i, batch_size=config.batch_size) for test_set_i in iso_full_sets]

# aniso_pad_sets = [MeshDataset(
#     os.path.join(data_path, "data"),
#     transform=normalise if config.is_normalise else None,
#     x_feature=config.x_feat,
#     mesh_feature=config.mesh_feat,
#     conv_feature=config.conv_feat,
#     conv_feature_fix=config.conv_feat_fix,
#     load_jacobian=config.use_jacob,
#     use_cluster=config.use_cluster,
#     r=config.cluster_r,
# ) for data_path in data_paths_aniso_pad]

# aniso_pad_loaders = [DataLoader(test_set_i, batch_size=config.batch_size) for test_set_i in aniso_pad_sets]

aniso_full_sets = [
    MeshDataset(
        os.path.join(data_path, "data"),
        transform=normalise if config.is_normalise else None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        load_jacobian=config.use_jacob,
        use_cluster=config.use_cluster,
        r=config.cluster_r,
    )
    for data_path in data_paths_aniso_full
]

aniso_full_loaders = [
    DataLoader(test_set_i, batch_size=config.batch_size)
    for test_set_i in aniso_full_sets
]


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


train_func = train_unsupervised
evaluate_func = evaluate_unsupervised
for epoch in range(config.num_epochs + 1):
    #   train_loss = train(train_loader, model, optimizer, device, loss_func=loss_func,
    #                      use_area_loss=config.use_area_loss,
    #                      scaler=300,
    #                      )
    #   test_loss = evaluate(test_loader, model, device, loss_func=loss_func,
    #                        use_area_loss=config.use_area_loss,
    #                        scaler=300,
    #                        )
    train_loss = train_func(
        train_loader,
        model,
        optimizer,
        device,
        loss_func=loss_func,
        use_area_loss=config.use_area_loss,
        use_convex_loss=config.use_convex_loss,
        weight_area_loss=config.weight_area_loss,
        weight_deform_loss=config.weight_deform_loss,
        weight_eq_residual_loss=config.weight_eq_residual_loss,
        scaler=300,
    )
    test_loss = evaluate_func(
        test_loader,
        model,
        device,
        loss_func=loss_func,
        use_area_loss=config.use_area_loss,
        use_convex_loss=config.use_convex_loss,
        weight_area_loss=config.weight_area_loss,
        weight_deform_loss=config.weight_deform_loss,
        weight_eq_residual_loss=config.weight_eq_residual_loss,
        scaler=300,
    )
    wandb.log(
        {
            "Deform Loss/Train": train_loss["deform_loss"],
            "Deform Loss/Test": test_loss["deform_loss"],
        },
        step=epoch,
    )
    wandb.log(
        {
            "Equation residual/Train": train_loss["equation_residual"],
            "Equation residual/Test": test_loss["equation_residual"],
        },
        step=epoch,
    )
    print(f"Epoch: {epoch}")
    if config.use_convex_loss:
        wandb.log(
            {
                "Convex loss/Train": train_loss["convex_loss"],
                "Convex loss/Test": test_loss["convex_loss"],
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

    if (epoch) % config.check_tangle_interval == 0:
        train_tangle = count_dataset_tangle(
            train_set, model, device, method=config.count_tangle_method
        )
        test_tangle = count_dataset_tangle(
            test_set, model, device, method=config.count_tangle_method
        )
        wandb.log(
            {
                "Tangled Elements per Mesh/Train": train_tangle,
                "Tangled Elements per Mesh/Test": test_tangle,
            },
            step=epoch,
        )
    # check loss and tangle for each mesh_size under different datasets:
    if (epoch) % config.multi_scale_check_interval == 0:
        # iso_pad_losses = [
        #     evaluate_func(
        #         loader_i, model, device, loss_func=loss_func,
        #         use_jacob=config.use_jacob)["deform_loss"] for loader_i in iso_pad_loaders
        #     ]
        # torch.cuda.empty_cache()
        # gc.collect()
        # iso_full_losses = [
        #     evaluate_func(
        #         loader_i, model, device, loss_func=loss_func,
        #         use_jacob=config.use_jacob)["deform_loss"] for loader_i in iso_full_loaders
        #     ]
        # torch.cuda.empty_cache()
        # gc.collect()
        # aniso_pad_losses = [
        #     evaluate_func(
        #         loader_i, model, device, loss_func=loss_func,
        #         use_jacob=config.use_jacob)["deform_loss"] for loader_i in aniso_pad_loaders
        #     ]
        # torch.cuda.empty_cache()
        # gc.collect()
        aniso_full_losses = [
            evaluate_func(
                loader_i, model, device, loss_func=loss_func, use_jacob=config.use_jacob
            )["deform_loss"]
            for loader_i in aniso_full_loaders
        ]
        torch.cuda.empty_cache()
        gc.collect()
        # iso_pad_tangle = [
        #     count_dataset_tangle(train_set_i, model, device, method=config.count_tangle_method) for train_set_i in iso_pad_sets
        #     ]
        # torch.cuda.empty_cache()
        # gc.collect()
        # iso_full_tangle = [
        #     count_dataset_tangle(train_set_i, model, device, method=config.count_tangle_method) for train_set_i in iso_full_sets
        #     ]
        # torch.cuda.empty_cache()
        # gc.collect()
        # aniso_pad_tangle = [
        #     count_dataset_tangle(train_set_i, model, device, method=config.count_tangle_method) for train_set_i in aniso_pad_sets
        #     ]
        # torch.cuda.empty_cache()
        # gc.collect()
        aniso_full_tangle = [
            count_dataset_tangle(
                train_set_i, model, device, method=config.count_tangle_method
            )
            for train_set_i in aniso_full_sets
        ]
        torch.cuda.empty_cache()
        gc.collect()
        for i in range(len(config.n_grids_test)):
            n_grids = config.n_grids_test[i]
            #   wandb.log({f"TEpM(iso_pad)/mesh_size:{n_grids}": iso_pad_tangle[i]}, step=epoch)
            #   wandb.log({f"TEpM(iso_full)/mesh_size:{n_grids}": iso_full_tangle[i]}, step=epoch)
            #   wandb.log({f"TEpM(aniso_pad)/mesh_size:{n_grids}": aniso_pad_tangle[i]}, step=epoch)
            wandb.log(
                {f"TEpM(aniso_full)/mesh_size:{n_grids}": aniso_full_tangle[i]},
                step=epoch,
            )

            #   wandb.log({f"Deform Loss(iso_pad)/mesh_size:{n_grids}": iso_pad_losses[i]}, step=epoch)
            #   wandb.log({f"Deform Loss(iso_full)/mesh_size:{n_grids}": iso_full_losses[i]}, step=epoch)
            #   wandb.log({f"Deform Loss(aniso_pad)/mesh_size:{n_grids}": aniso_pad_losses[i]}, step=epoch)
            wandb.log(
                {f"Deform Loss(aniso_full)/mesh_size:{n_grids}": aniso_full_losses[i]},
                step=epoch,
            )

    if (epoch + 1) % config.save_interval == 0:
        torch.save(model.state_dict(), "{}/model_{}.pth".format(output_folder, epoch))
        # artifact.add_file(local_path="model_{}.pth".format(epoch))
        wandb.save("{}/model_{}.pth".format(output_folder, epoch))
# run.log_artifact(artifact)
run.finish()

# runtime.unassign()
