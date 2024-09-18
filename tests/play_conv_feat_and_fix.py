# auther: Chunyang Wang
# Github account: chunyang-w

# This file plot the conv_feat and conv_feat_fix for a given
# training sample, to see if the conv_feat_fix and conv_feat are similar.

import warnings

import matplotlib.pyplot as plt
import torch

import warpmesh as wm

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# put the path to your data here
test_data_dir = "/Users/cw1722/Documents/warpmesh/data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_<15x15>_n=10_aniso/train"  # noqa

idx = 3

# features to load
conv_feat_fix = [
    "conv_uh_fix",
    "conv_hessian_norm_fix",
]

conv_feat = [
    "conv_uh",
    "conv_hessian_norm",
]

data_set = wm.MeshDataset(
    test_data_dir,
    conv_feature_fix=conv_feat_fix,
    conv_feature=conv_feat,
    transform=wm.normalise,
)

sample = data_set[idx]


print("sample:", sample)
print("conv_feat_fix.shape", sample.conv_feat_fix.shape)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].set_title("conv_feat 0")
axs[0, 1].set_title("conv_feat 1")
axs[1, 0].set_title("conv_feat_fix 0")
axs[1, 1].set_title("conv_feat_fix 1")

im00 = axs[0, 0].imshow(sample.conv_feat[0, :, :].cpu().numpy())
im01 = axs[0, 1].imshow(sample.conv_feat[1, :, :].cpu().numpy())

im10 = axs[1, 0].imshow(sample.conv_feat_fix[0, :, :].cpu().numpy())
im11 = axs[1, 1].imshow(sample.conv_feat_fix[1, :, :].cpu().numpy())

fig.colorbar(im00, ax=axs[0, 0], fraction=0.046, pad=0.04)
fig.colorbar(im01, ax=axs[0, 1], fraction=0.046, pad=0.04)
fig.colorbar(im10, ax=axs[1, 0], fraction=0.046, pad=0.04)
fig.colorbar(im11, ax=axs[1, 1], fraction=0.046, pad=0.04)

plt.show()
