# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# Code for training the model locally. May require modification to run this.

import os
import warnings
from datetime import datetime

import pandas as pd
import torch
from rich.console import Console
from rich.live import Live
from rich.progress import Progress
from torch_geometric.data import DataLoader

import UM2N

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_mse(out, data, weight):
    sq_diff = (out - data.y) ** 2
    weight = weight.unsqueeze(1)
    w_mse = torch.mean(weight * sq_diff)
    return w_mse


#  ================PARAMETERS====================================
problem = "helmholtz"

use_jacob = True

# model = UM2N.M2N(gfe_in_c=2, lfe_in_c=4)
model = UM2N.MRN(gfe_in_c=2, lfe_in_c=4)

weight_decay = 5e-4
train_batch_size = 20

pre_train_weight_path = None

simple_u = True
data_type = "smpl" if simple_u else "cmplx"

n_dist = None
max_dist = 6
n_elem_x = 20
n_elem_y = 20
num_samples = 400

num_epochs = 2
print_interval = 1
save_interval = 1

loss_func = torch.nn.L1Loss()
# loss_func = torch.nn.MSELoss()

normalise = True
# normalise = False

x_feat = [
    "coord",
    "bd_mask",
    "bd_left_mask",
    "bd_right_mask",
    "bd_down_mask",
    "bd_up_mask",
]
mesh_feat = [
    "coord",
    "u",
    "hessian_norm",
    # "grad_u",
    # "hessian",
]
conv_feat = [
    "conv_uh",
    "conv_hessian_norm",
]

project_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
data_set_path = os.path.join(project_dir, "data/dataset/helmholtz")
data_path = os.path.join(
    data_set_path,
    "z=<0,1>_ndist={}_max_dist={}_<{}x{}>_n={}_{}/".format(
        n_dist, max_dist, n_elem_x, n_elem_y, num_samples, data_type
    ),
)
#  ==============================================================


#  ================BUILD DIR====================================
print("Project directory: ", project_dir)

now = datetime.now()
output_dir = os.path.join(project_dir, "data", "output", now.strftime("%Y-%m-%d %H:%M"))

prediction_dir = os.path.join(output_dir, "prediction")
trainlog_dir = os.path.join(output_dir, "train_log")
weight_dir = os.path.join(output_dir, "weight")

UM2N.mkdir_if_not_exist(prediction_dir)
UM2N.mkdir_if_not_exist(trainlog_dir)
UM2N.mkdir_if_not_exist(weight_dir)

df = pd.DataFrame(
    {
        "Problem": [problem],
        "Model": [model],
        "n_dist": [n_dist],
        "max_dist": [max_dist],
        "n_elem_x": [n_elem_x],
        "n_elem_y": [n_elem_y],
        "num_samples": [num_samples],
        "data_path": [data_path],
        "num_epochs": [num_epochs],
        "print_interval": [print_interval],
        "loss_func": [loss_func],
        "normalise": [normalise],
        "x_feature": [x_feat],
        "mesh_feature": [mesh_feat],
        "conv_feat": [conv_feat],
        "pre_train_weight_path": [pre_train_weight_path],
        "simple_u": [simple_u],
        "weight_decay": [weight_decay],
    }
)

# convert to a tall table
df = df.melt(var_name="Training params", value_name="Value")

# write to csv
df.to_csv(os.path.join(output_dir, "info.csv"), index=False)
#  ==============================================================


#  ================LOAD DATA====================================
train_set = UM2N.MeshDataset(
    os.path.join(data_path, "train"),
    transform=UM2N.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)
test_set = UM2N.MeshDataset(
    os.path.join(data_path, "test"),
    transform=UM2N.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)
val_set = UM2N.MeshDataset(
    os.path.join(data_path, "val"),
    transform=UM2N.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)

train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10)
val_loader = DataLoader(val_set, batch_size=10)

len(train_set), len(test_set), len(val_set)
#  ==============================================================


#  ================TRAIN FUNCs=================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)

model = model.to(device)

if pre_train_weight_path is not None:
    state_dict = torch.load(pre_train_weight_path)
    model.load_state_dict(state_dict)


#  ================Progress Bar==================================
progress = Progress()
console = Console()
task = progress.add_task("[cyan]Training...", total=num_epochs)
# ===============================================================


#  =================TRAIN=======================================
if __name__ == "__main__":
    train_loss_arr = []
    test_loss_arr = []
    train_tangle_arr = []
    test_tangle_arr = []
    epoch_arr = []
    with Live(progress, refresh_per_second=5):  # update 10 times a second
        for epoch in range(num_epochs):
            progress.update(task, advance=1)

            train_loss = UM2N.train(
                train_loader,
                model,
                optimizer,
                device,
                loss_func=loss_func,
                use_jacob=use_jacob,
            )
            test_loss = UM2N.evaluate(
                test_loader, model, device, loss_func=loss_func, use_jacob=use_jacob
            )

            train_tangle = UM2N.check_dataset_tangle(train_set, model, n_elem_x, n_elem_y)
            test_tangle = UM2N.check_dataset_tangle(test_set, model, n_elem_x, n_elem_y)

            train_loss_arr.append(train_loss)
            test_loss_arr.append(test_loss)
            train_tangle_arr.append(train_tangle)
            test_tangle_arr.append(test_tangle)
            epoch_arr.append(epoch)

            if (epoch) % print_interval == 0:
                print(
                    f"\nEpoch: {epoch+1}/{num_epochs}, "
                    f"\nTrain Loss: {train_loss:.4f}, "
                    f"\nTest Loss: {test_loss:.4f}, "
                    f"\nTrain Tangle: {train_tangle:.4f}, "
                    f"\nTest Tangle: {test_tangle:.4f}\n"
                )
                print("node inspect:")
                out = model(val_set[0].to(device))
                print(out[:10, :])
                print("=====================================")
            if (epoch + 1) % save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(weight_dir, "model_{}.pth".format(epoch)),
                )

    # plot prediction
    UM2N.plot_prediction(
        val_set,
        model,
        prediction_dir,
        mode="val",
        loss_fn=loss_func,
        n_elem_x=n_elem_x,
        n_elem_y=n_elem_y,
    )
    UM2N.plot_prediction(
        test_set,
        model,
        prediction_dir,
        mode="test",
        loss_fn=loss_func,
        n_elem_x=n_elem_x,
        n_elem_y=n_elem_y,
    )

    # plot traing curve
    fig = UM2N.plot_loss(train_loss_arr, test_loss_arr, epoch_arr)
    fig.savefig(os.path.join(trainlog_dir, "loss.png"))

    # plot avg tangle curve
    fig = UM2N.plot_tangle(train_tangle_arr, test_tangle_arr, epoch_arr)
    fig.savefig(os.path.join(trainlog_dir, "tangle.png"))

    # write final loss
    df_train = pd.DataFrame(
        {
            "Epoch": epoch_arr,
            "Train Loss": train_loss_arr,
            "Test Loss": test_loss_arr,
        }
    )
    df_train.to_csv(os.path.join(trainlog_dir, "loss.csv"), index=False)

    df_tangle = pd.DataFrame(
        {
            "Epoch": epoch_arr,
            "Train tangle": train_tangle_arr,
            "Test tangle": test_tangle_arr,
        }
    )
    df_tangle.to_csv(os.path.join(trainlog_dir, "tangle.csv"), index=False)

    # mark the folder if done
    with open(os.path.join(output_dir, "done"), "w") as f:
        f.write("Done")
# ==========================================================
