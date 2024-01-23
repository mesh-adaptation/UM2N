# package import
# %load_ext autoreload
# %autoreload 2

from warpmesh.model import M2N, train, train_unsupervised, evaluate, evaluate_unsupervised, MRN, count_dataset_tangle, M2N_dynamic_drop, M2N_dynamic_no_drop, MRNAtten, M2NAtten
from warpmesh.model import MRNGlobalTransformerEncoder, MRNLocalTransformerEncoder, MRTransformer, M2Transformer
from warpmesh.helper import mkdir_if_not_exist, plot_loss, plot_tangle
from warpmesh.helper import save_namespace_to_yaml, load_yaml_to_namespace
from warpmesh.loader import MeshDataset, normalise, AggreateDataset
# from google.colab import runtime
import os
import gc
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import warnings
import warpmesh as wm
from torch_geometric.data import DataLoader
from IPython import display
import wandb
# from google.colab import userdata
import argparse
from argparse import Namespace
from types import SimpleNamespace

# random_seed = 666
# torch.manual_seed(random_seed)


parser = argparse.ArgumentParser(
                    prog='Warpmesh',
                    description='warp the mesh',
                    epilog='warp the mesh')
parser.add_argument('-config', default='', type=str, required=True)
args = parser.parse_args()


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb.login(key=userdata.get("wandb_key"))
wandb.login(key="9e49ed1812a0349724515be9c3c856f4b1c86cad")

config_name = args.config
config = load_yaml_to_namespace(f"./configs/{config_name}")

# Define path where data get stored
now = datetime.now()
now_date = now.strftime("%Y-%m-%d-%H:%M_")
config.experiment_name = now_date + config_name

model = None
if (config.model_used == "MRTransformer"):
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
    device=device
)
else:
    raise Exception(f"Model {config.model_used} not implemented.")

# =================== load from checkpoint ==========================

# Load from checkpoint
entity = 'mz-team'
project_name = 'warpmesh'
run_id = '8ndi2teh'
api = wandb.Api()
run_loaded = api.run(f"{entity}/{project_name}/{run_id}")
epoch = 999
target_file_name = "model_{}.pth".format(epoch)
model_file = None
model_store_path = './fine_tune_model'
for file in run_loaded.files():
    if file.name.endswith(target_file_name):
        model_file = file.download(root=model_store_path, replace=True)
        target_file_name = file.name
assert model_file is not None, "Model file not found"
model_file_path = os.path.join(model_store_path, target_file_name)
model = wm.load_model(model, model_file_path)
print("Model checkpoint loaded.")
# ===================================================================


############### Change This To Dataset folder #################
data_root = config.data_root

# data set for training
data_paths = [f"{config.data_root}z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2"]
###############################################################


loss_func = torch.nn.L1Loss()

print(model)
print()
print(data_paths)

# Load datasets
train_sets = [MeshDataset(
    os.path.join(data_path, "train"),
    transform=normalise if config.is_normalise else None,
    x_feature=config.x_feat,
    mesh_feature=config.mesh_feat,
    conv_feature=config.conv_feat,
    conv_feature_fix=config.conv_feat_fix,
    load_jacobian=config.use_jacob,
    use_cluster=config.use_cluster,
    r=config.cluster_r,
) for data_path in data_paths]

test_sets = [MeshDataset(
    os.path.join(data_path, "test"),
    transform=normalise if config.is_normalise else None,
    x_feature=config.x_feat,
    mesh_feature=config.mesh_feat,
    conv_feature=config.conv_feat,
    conv_feature_fix=config.conv_feat_fix,
    load_jacobian=config.use_jacob,
    use_cluster=config.use_cluster,
    r=config.cluster_r,
) for data_path in data_paths]

val_sets = [MeshDataset(
    os.path.join(data_path, "val"),
    transform=normalise if config.is_normalise else None,
    x_feature=config.x_feat,
    mesh_feature=config.mesh_feat,
    conv_feature=config.conv_feat,
    conv_feature_fix=config.conv_feat_fix,
    load_jacobian=config.use_jacob,
    use_cluster=config.use_cluster,
    r=config.cluster_r,
) for data_path in data_paths]

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
    project = config.project,
    name = config.experiment_name,
    tags = [config.model_used],
    config = config.__dict__,
)
# artifact = wandb.Artifact(name=config.experiment_name.replace(':', '_'), type="model")


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate,
    betas=(0.9, 0.999), eps=1e-08,
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
  
  train_loss = train_func(train_loader, model, optimizer, device, loss_func=loss_func,
                     use_area_loss=config.use_area_loss,
                     use_convex_loss=config.use_convex_loss,
                     weight_area_loss=config.weight_area_loss,
                     weight_deform_loss=config.weight_deform_loss,
                     weight_eq_residual_loss=config.weight_eq_residual_loss,
                     scaler=300,
                     )
  test_loss = evaluate_func(test_loader, model, device, loss_func=loss_func,
                       use_area_loss=config.use_area_loss,
                       use_convex_loss=config.use_convex_loss,
                       weight_area_loss=config.weight_area_loss,
                       weight_deform_loss=config.weight_deform_loss,
                       weight_eq_residual_loss=config.weight_eq_residual_loss,
                       scaler=300,
                       )
  wandb.log({
      "Deform Loss/Train": train_loss["deform_loss"],
      "Deform Loss/Test":test_loss["deform_loss"],
  }, step=epoch)
#   wandb.log({
#       "Boundary Loss/Train": train_loss["boundary_loss"],
#       "Boundary Loss/Test":test_loss["boundary_loss"],
#   }, step=epoch)
  wandb.log({
        "Equation residual/Train": train_loss["equation_residual"],
        "Equation residual/Test":test_loss["equation_residual"],
  }, step=epoch)
  print(f"Epoch: {epoch}")
  if (config.use_convex_loss):
      wandb.log({
        "Convex loss/Train": train_loss["convex_loss"],
        "Convex loss/Test":test_loss["convex_loss"],
      }, step=epoch)
  if (config.use_inversion_loss):
      wandb.log({
          "Inversion Loss/Train": train_loss["inversion_loss"],
          "Inversion Loss/Test":test_loss["inversion_loss"],
      }, step=epoch)
  if (config.use_area_loss):
      wandb.log({
          "Area Loss/Train": train_loss["area_loss"],
          "Area Loss/Test":test_loss["area_loss"],
      }, step=epoch)

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

# runtime.unassign()