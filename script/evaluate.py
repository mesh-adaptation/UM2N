# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os
import re
import glob
import torch
import warnings
import pandas as pd
from torch_geometric.data import DataLoader
os.environ["OMP_NUM_THREADS"] = "1"
import warpmesh as wm  # noqa

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  =================================================================
out_dir = '/Users/cw1722/Documents/irp/irp-cw1722/data/output/2023-07-29 17_52'  # noqa
data_dir = '/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<20x20>_n=400_smpl'  # noqa

model = wm.MRN(gfe_in_c=2, lfe_in_c=4)

loss_func = torch.nn.L1Loss()

n_elem_x = n_elem_y = 20

normalise = True

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
#  =================================================================

weight_dir = os.path.join(out_dir, "weight")
train_log_dir = os.path.join(out_dir, "train_log")
prediction_dir = os.path.join(out_dir, "prediction")

weight_path = os.path.join(weight_dir, 'model_*.pth')
weight_name = glob.glob(weight_path)

train_set = wm.MeshDataset(
    os.path.join(data_dir, "train"),
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)
test_set = wm.MeshDataset(
    os.path.join(data_dir, "test"),
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)
val_set = wm.MeshDataset(
    os.path.join(data_dir, "val"),
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)

train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10)
val_loader = DataLoader(val_set, batch_size=10)


def extract_epoch(filepath):
    match = re.search(r'model_(\d+)', filepath)
    if match:
        epoch = int(match.group(1))
        return epoch
    else:
        return 0


weight_arr = sorted(weight_name, key=extract_epoch)


if __name__ == "__main__":
    train_loss_arr = []
    test_loss_arr = []
    epoch_arr = []
    tangle_test_arr = []
    tangle_train_arr = []
    for i in range(len(weight_arr)):
        match = re.search(r'model_(\d+)', weight_arr[i])
        epoch = int(match.group(1))
        model = wm.load_model(model, weight_arr[i])
        # train_loss = wm.evaluate(train_loader,
        #                          model, device, loss_func)
        # test_loss = wm.evaluate(test_loader,
        #                         model, device, loss_func)
        # train_tangle = wm.check_dataset_tangle(
        #     train_set, model, n_elem_x, n_elem_y)
        test_tangle = wm.check_dataset_tangle(
            test_set, model, n_elem_x, n_elem_y)
        train_loss = 0
        test_loss = 0
        train_tangle = 0
        # test_tangle = 0
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss}")
        print(f"Test Loss: {test_loss}")
        print(f"Train Tangle: {train_tangle}")
        print(f"Test Tangle: {test_tangle}")
        print("=====================================")
        print()

        tangle_test_arr.append(test_tangle)
        tangle_train_arr.append(train_tangle)
        train_loss_arr.append(train_loss)
        test_loss_arr.append(test_loss)
        epoch_arr.append(epoch)

    # plot the loss

    # plot prediction
    wm.plot_prediction(val_set, model, prediction_dir,
                       mode="val", loss_fn=loss_func,
                       n_elem_x=n_elem_x, n_elem_y=n_elem_y)
    wm.plot_prediction(test_set, model, prediction_dir,
                       mode="test", loss_fn=loss_func,
                       n_elem_x=n_elem_x, n_elem_y=n_elem_y)

    # plot traing curve
    fig = wm.plot_loss(train_loss_arr, test_loss_arr, epoch_arr)
    # fig.savefig(os.path.join(train_log_dir, "loss.png"))

    # plot avg tangle curve
    fig = wm.plot_tangle(tangle_train_arr, tangle_test_arr, epoch_arr)
    fig.savefig(os.path.join(train_log_dir, "tangle.png"))

    # write final loss
    df_train = pd.DataFrame({
        'Epoch': epoch_arr,
        'Train Loss': train_loss_arr,
        'Test Loss': test_loss_arr,
    })
    df_train.to_csv(
        os.path.join(train_log_dir, 'loss.csv'),
        index=False)

    df_tangle = pd.DataFrame({
        'Epoch': epoch_arr,
        'Train tangle': tangle_train_arr,
        'Test tangle': tangle_test_arr,
    })
    df_tangle.to_csv(
        os.path.join(train_log_dir, 'tangle.csv'),
        index=False)

    # mark the folder if done
    with open(os.path.join(out_dir, "done"), "w") as f:
        f.write("Evaluated")
