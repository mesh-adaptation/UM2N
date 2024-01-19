# Author: Chunyang Wang
# GitHub Username: chunyang-w

# import packages
import datetime
import glob
import time
import torch
import os
import wandb

import firedrake as fd
import matplotlib.pyplot as plt
import pandas as pd
import warpmesh as wm

from torch_geometric.loader import DataLoader
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

project_name = 'warpmesh'
# run_id = 'sr7waaso'  # MRT with no mask
# run_id = 'gl1zpjc5'  # MRN 3-loop
run_id = '3wv8mgyt'  # MRN 3-loop, on polymesh

epoch = 599
ds_root = (  # square
        '/Users/chunyang/projects/WarpMesh/data/dataset/helmholtz/'
        'z=<0,1>_ndist=None_max_dist=6_<25x25>_n=100_aniso_full')