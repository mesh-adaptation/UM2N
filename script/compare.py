# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# %% import block
import warnings

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd
import torch
from torch_geometric.data import DataLoader  # noqa

import UM2N

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.no_grad()

# %% load model
n_elem = 20
data_dir = f"/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{n_elem}x{n_elem}>_n=400_cmplx/val"  # noqa

M2N_weight_path = "/Users/cw1722/Downloads/M2N__15,20__cmplx/weight/model_999.pth"  # noqa
# MRN_path = "/Users/cw1722/Downloads/MRN_r=5_15,20__smpl/weight/model_999.pth"  # noqa
MRN_path = "/Users/cw1722/Downloads/MRN_r=5__15,20__cmplx/weight/model_999.pth"  # noqa

model_M2N = UM2N.M2N(
    deform_in_c=7,
    gfe_in_c=2,
    lfe_in_c=4,
).to(device)
model_M2N = UM2N.load_model(model_M2N, M2N_weight_path)

model_MRN = UM2N.MRN(
    deform_in_c=7,
    gfe_in_c=2,
    lfe_in_c=4,
    num_loop=5,
).to(device)
model_M2N.eval()
model_MRN = UM2N.load_model(model_MRN, MRN_path)
model_MRN.eval()

# %% dataset load
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
normalise = True
loss_func = torch.nn.L1Loss()
data_set = UM2N.MeshDataset(
    data_dir,
    transform=UM2N.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
    load_analytical=True,
)

# %% test on validation set to calculate tangle, speedup, loss.


def compare_on_dataset(model, dataset):
    tangle = []
    error_og = []
    error_ma = []
    error_model = []
    time_ma = []
    time_model = []
    acceleration = []
    for data_idx in range(len(dataset)):
        data = dataset[data_idx]
        res = UM2N.compare_error(model, data, plot=True, n_elem=n_elem)
        tangle.append(res["tangle_num"])
        error_og.append(res["error_original_mesh"])
        error_ma.append(res["error_ma_mesh"])
        error_model.append(res["error_model_mesh"])
        time_ma.append(res["time_ma"])
        time_model.append(res["time_model"])
        acceleration.append(res["acceleration"])
        plt.show()
    return {
        "tangle": tangle,
        "error_og": error_og,
        "error_ma": error_ma,
        "error_model": error_model,
        "time_ma": time_ma,
        "time_model": time_model,
        "acceleration": acceleration,
    }


# %%
res_m2n = compare_on_dataset(model_M2N, data_set)
res_mrn = compare_on_dataset(model_MRN, data_set)


# %%


def remove_elems(res):
    invalid = []
    large_error = []
    for i in range(len(res["error_og"])):
        if res["error_ma"][i] > res["error_og"][i]:
            large_error.append(i)
        if res["error_model"][i] is None:
            invalid.append(i)
    # perform a set opperation
    remove_idx = set(invalid + large_error)
    for key in res.keys():
        res[key] = np.delete(np.array(res[key]), list(remove_idx))
    return res


# %% compare info summary
print(res_m2n)
print(res_mrn)

# %% compute statictics and log
res = res_m2n
tangle = res["tangle"]
error_og = res["error_og"]
error_ma = res["error_ma"]
error_model = res["error_model"]
time_ma = res["time_ma"]
time_model = res["time_model"]
acceleration = res["acceleration"]

tangle = np.mean(np.array(tangle))
error_reduction_ma = np.mean(
    (np.array(error_og) - np.array(error_ma)) / np.array(error_og)
)  # noqa
error_reduction_model = np.mean(
    (np.array(error_og) - np.array(error_model)) / np.array(error_og)
)  # noqa

print("M2N")
print(f"tangle: {tangle}")
print(f"error_reduction_ma: {error_reduction_ma}")
print(f"error_reduction_model: {error_reduction_model}")
print(f"acceration: {np.mean(np.array(acceleration))}")

# %% compute statictics and log
res = res_mrn
tangle = res["tangle"]
error_og = res["error_og"]
error_ma = res["error_ma"]
error_model = res["error_model"]
time_ma = res["time_ma"]
time_model = res["time_model"]
acceleration = res["acceleration"]

tangle = np.mean(np.array(tangle))
error_reduction_ma = np.mean(
    (np.array(error_og) - np.array(error_ma)) / np.array(error_og)
)  # noqa
error_reduction_model = np.mean(
    (np.array(error_og) - np.array(error_model)) / np.array(error_og)
)  # noqa
print("MRN")
print(f"tangle: {tangle}")
print(f"error_reduction_ma: {error_reduction_ma}")
print(f"error_reduction_model: {error_reduction_model}")
print(f"acceration: {np.mean(np.array(acceleration))}")

# %% plot sample from m2n
data = data_set[2]
UM2N.compare_error(model_MRN, data, plot=True, n_elem=n_elem)

# %% plot model training loss, test tangle
loss_m2n_path = "/Users/cw1722/Downloads/M2N__15,20__cmplx/train_log/loss.csv"
loss_mrn_path = "/Users/cw1722/Downloads/MRN_r=5__15,20__cmplx/train_log/loss.csv"  # noqa
tangle_m2n_path = "/Users/cw1722/Downloads/M2N__15,20__cmplx/train_log/tangle.csv"  # noqa
tangle_mrn_path = "/Users/cw1722/Downloads/MRN_r=5__15,20__cmplx/train_log/tangle.csv"  # noqa

mrn_loss_data = pd.read_csv(loss_mrn_path)
m2n_loss_data = pd.read_csv(loss_m2n_path)
tangle_m2n_data = pd.read_csv(tangle_m2n_path)
tangle_mrn_data = pd.read_csv(tangle_mrn_path)

mrn_loss_data_filtered = mrn_loss_data[
    (mrn_loss_data["Train Loss"] <= 50) & (mrn_loss_data["Test Loss"] <= 50)
]  # noqa
m2n_loss_data_filtered = m2n_loss_data[
    (m2n_loss_data["Train Loss"] <= 50) & (m2n_loss_data["Test Loss"] <= 50)
]  # noqa
tangle_m2n_data_filtered = tangle_m2n_data[tangle_m2n_data["Test Tangle"] <= 10]  # noqa
tangle_mrn_data_filtered = tangle_mrn_data[tangle_mrn_data["Test Tangle"] <= 10]  # noqa

# Combined subplot with loss on the left and tangle on the right
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7))

# Left subplot: Loss plot
ax1.plot(
    m2n_loss_data_filtered["Epoch"],
    m2n_loss_data_filtered["Train Loss"],
    label="M2N Train Loss",
    color="cyan",
    lw=2,
    alpha=0.65,
)  # noqa
ax1.plot(
    m2n_loss_data_filtered["Epoch"],
    m2n_loss_data_filtered["Test Loss"],
    label="M2N Test Loss",
    color="blue",
    lw=2,
    linestyle="--",
    alpha=0.65,
)  # noqa
ax1.plot(
    mrn_loss_data_filtered["Epoch"],
    mrn_loss_data_filtered["Train Loss"],
    label="MRN Train Loss",
    color="red",
    lw=2,
    alpha=0.65,
)  # noqa
ax1.plot(
    mrn_loss_data_filtered["Epoch"],
    mrn_loss_data_filtered["Test Loss"],
    label="MRN Test Loss",
    color="orange",
    lw=2,
    linestyle="--",
    alpha=0.65,
)  # noqa
final_epoch_m2n = m2n_loss_data_filtered["Epoch"].iloc[-1]
final_test_loss_m2n = m2n_loss_data_filtered["Test Loss"].iloc[-1]
final_epoch_mrn = mrn_loss_data_filtered["Epoch"].iloc[-1]
final_test_loss_mrn = mrn_loss_data_filtered["Test Loss"].iloc[-1]
ax1.annotate(
    f"M2N: {final_test_loss_m2n:.2f}",
    (final_epoch_m2n, final_test_loss_m2n - 3),
    textcoords="offset points",
    xytext=(0, 0),
    ha="center",
    fontsize=20,
    color="blue",
    weight="bold",
)  # noqa
ax1.annotate(
    f"MRN: {final_test_loss_mrn:.2f}",
    (final_epoch_mrn, final_test_loss_mrn + 3),
    textcoords="offset points",
    xytext=(0, 0),
    ha="center",
    fontsize=20,
    color="orange",
    weight="bold",
)  # noqa
ax1.set_xlabel("Epoch", fontsize=22)
ax1.set_ylabel("Loss", fontsize=22)
ax1.legend(loc="upper right", fontsize=20)
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis="both", which="major", labelsize=20)

# Right subplot: Tangle plot
ax2.plot(
    tangle_m2n_data_filtered["Epoch"],
    tangle_m2n_data_filtered["Test Tangle"],
    label="M2N Test Tangle",
    color="blue",
    lw=2,
)  # noqa
ax2.plot(
    tangle_mrn_data_filtered["Epoch"],
    tangle_mrn_data_filtered["Test Tangle"],
    label="MRN Test Tangle",
    color="orange",
    lw=2,
)  # noqa
final_epoch_m2n_tangle = tangle_m2n_data_filtered["Epoch"].iloc[-1]
final_test_tangle_m2n = tangle_m2n_data_filtered["Test Tangle"].iloc[-1]
final_epoch_mrn_tangle = tangle_mrn_data_filtered["Epoch"].iloc[-1]
final_test_tangle_mrn = tangle_mrn_data_filtered["Test Tangle"].iloc[-1]
ax2.annotate(
    f"M2N: {final_test_tangle_m2n:.4f}",
    (final_epoch_m2n_tangle, final_test_tangle_m2n + 0.07),
    textcoords="offset points",
    xytext=(10, 0),
    ha="center",
    fontsize=20,
    color="blue",
    weight="bold",
)  # noqa
ax2.annotate(
    f"MRN: {final_test_tangle_mrn:.4f}",
    (final_epoch_mrn_tangle, final_test_tangle_mrn + 0.02),
    textcoords="offset points",
    xytext=(10, 0),
    ha="center",
    fontsize=20,
    color="orange",
    weight="bold",
)  # noqa
ax2.set_ylim(0, 1)
ax2.set_xlabel("Epoch", fontsize=22)
ax2.set_ylabel("Test Tangle", fontsize=22)
ax2.legend(loc="upper right", fontsize=20)
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(axis="both", which="major", labelsize=20)

# Adjust the layout and display the combined subplot
plt.tight_layout()
plt.show()

# %% test on sets for loss and tangle


def summrise_info(res):
    error_reduction_ma = np.mean((res["error_og"] - res["error_ma"]) / res["error_og"])
    error_reduction_model = np.mean(
        (res["error_og"] - res["error_model"]) / res["error_og"]
    )
    tangle = np.mean(res["tangle"])
    speedup = np.mean(res["acceleration"])
    return {
        "error_reduction_ma": error_reduction_ma,
        "error_reduction_model": error_reduction_model,
        "tangle": tangle,
        "speedup": speedup,
    }


dataset_list = [
    f"/Users/cw1722/Documents/irp/irp-cw1722/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{n_node}x{n_node}>_n=40_cmplx/train"
    for n_node in [18, 19, 20, 21, 22]  # noqa
]

data_sets = [
    UM2N.MeshDataset(
        data_dir,
        transform=UM2N.normalise if normalise else None,
        x_feature=x_feat,
        mesh_feature=mesh_feat,
        conv_feature=conv_feat,
        load_analytical=True,
    )
    for data_dir in dataset_list
]

res_list = []

for i in range(len(data_sets)):
    n_node = [18, 19, 20, 21, 22]
    ds = data_sets[i]
    n_elem = n_node[i]
    res = compare_on_dataset(model_MRN, ds)
    res = remove_elems(res)
    res_list.append(res)

for res in res_list:
    print()
    print(summrise_info(res))

# %%
