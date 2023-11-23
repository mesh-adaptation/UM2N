# Author: Chunyang Wang
# Github: chunyang-w

# In this file, we want to add extra edges within a range. All modification
# should be made in a 'in place' fashion. So disk space is not a concern.

# We need these functionalities:

#    1. Iterate through all the files in a directory, 'train', 'test' and 'val'
#    2. For each file, we need to read the file, and add extra edges

import os
import sys
import glob
import torch
import numpy as np
from argparse import ArgumentParser
cur_dir = os.path.dirname(__file__)
sys.path.append(cur_dir)
from cluster_utils import get_new_edges  # noqa


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument(
        '--target', type=str, default=None,
        help=(
                ('target directory. This dir should contain '),
                ('`train`, `test` and `val` subdirs.')
            )
    )
    parser.add_argument(
        '--r', type=float, default=0.35,
        help="radius of a cluster"
    )
    parser.add_argument(
        '--M', type=int, default=None,
        help="nodes in a cluster"
    )
    parser.add_argument(
        '--dist_weight', type=bool, default=False,
        help=(
                "use weighted probability to sample " +
                "nodes (according to distance to source)"
        )
    )
    parser.add_argument(
        '--add_nei', type=bool, default=False,
        help=(
                "add original neighbors to the cluster"
        )
    )
    args_ = parser.parse_args()
    print(args_)
    return args_


def add_edges(file_path, r, M, dist_weight, add_nei):
    """
    Add extra edges to the file
    1. Read the file
        1.1 get num_nodes
        1.2 get x
        1.3 get original edge_index
        1.4 get
    2. Add extra edges
    3. Save the file
    """
    # read in data
    data = np.load(file_path, allow_pickle=True)
    data_object = data.item()
    coords = torch.from_numpy(
        data_object.get('coord')
    )
    num_nodes = coords.shape[0]
    edge_index = torch.from_numpy(
        data_object.get('edge_index_bi')
    ).to(torch.int64)
    new_edges = get_new_edges(
        num_nodes, coords,
        edge_index, r, M,
        dist_weight, add_nei)
    data_object["cluster_edges"] = new_edges
    # save the file
    np.save(file_path, data_object)
    return


def process_subset(file_path, r, M, dist_weight, add_nei):
    file_pattern = os.path.join(file_path, 'data_*.npy')
    files = glob.glob(file_pattern)
    # print("files: ", files)
    print(f"processing {len(files)} files in{file_path}")
    for file in files:
        add_edges(file, r, M, dist_weight, add_nei)
    return


if __name__ == "__main__":
    print("Processing the dataset...")
    # define all the subdirectories
    all_folders = [
        'data', 'test', 'train', 'val'
    ]
    # parse arguments, get the target directory and cluster radius
    args_ = arg_parse()
    dataset_root = args_.target
    r = args_.r
    M = args_.M
    # dist_weight = True if args_.dist_weight == "True" else False
    # add_nei = True if args_.add_nei == "True" else False
    dist_weight = args_.dist_weight
    add_nei = args_.add_nei
    # get all the subdirectories
    subsets_path = [
        os.path.join(dataset_root, folder) for folder in all_folders
    ]
    # iterate through all the subsets
    for i in range(len(subsets_path)):
        process_subset(subsets_path[i], r, M, dist_weight, add_nei)
