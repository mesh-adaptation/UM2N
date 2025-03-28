# Author: Chunyang Wang
# GitHub Username: chunyang-w

# Functionality: Evaluate the model performance on the given dataset
#                   + consume time
#                   + error reduction
#                   + tangled element per mesh

# Input:
#       + model path
#       + dataset path
# Output:
#       + evaluation results


# %% import packages and setup
import warnings

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch
from torch_geometric.data import DataLoader  # noqa

import UM2N

torch.no_grad()
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_weight_path = "/Users/cw1722/Downloads/M2N__15,20__cmplx/weight/model_999.pth"  # noqa

normalise = True  # normalise the input data or not
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
    # "conv_hessian_norm",
]

# %% load model
model = UM2N.M2N_og(
    deform_in_c=7,
    gfe_in_c=1,
    lfe_in_c=3,
).to(device)
model = UM2N.load_model(model, model_weight_path)
model.eval()
