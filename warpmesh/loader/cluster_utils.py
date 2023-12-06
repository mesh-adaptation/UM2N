import torch
from numba import njit
from torch_geometric.utils import (
    mask_to_index, index_to_mask
)

__all__ = ['sampler', 'get_neighbors', 'calc_dist',
           'get_new_edges',]


# vectorize version
def get_neighbors(source_mask, edge_idx):
    """
    Get the neighbors of the source nodes
    Args:
        data: the data object
        source_mask: a mask of the source nodes
        edge_idx: the edge index
    return:
        nei_mask: a mask of the neighbors
    """
    node_num = source_mask.shape[0]
    source_edges = source_mask[edge_idx[0]]

    target_nodes = edge_idx[1][source_edges]

    nei_mask = torch.zeros(node_num, dtype=torch.bool)
    nei_mask.scatter_(0, target_nodes, True)

    return nei_mask


def calc_dist(coords, node_idx, neighbors_mask):
    """
    Calculate the distance between the node and its neighbors
    Args:
        coords: the coordinates of the nodes
        node_idx: the index of the node
        neighbors_mask: a mask of the neighbors
    return:
        dist: the distance between the node and its neighbors
    """
    node_coords = coords[node_idx]
    nei_coords = coords[neighbors_mask]
    dist = torch.linalg.vector_norm(
        nei_coords - node_coords, dim=1
    )
    return dist


# def sampler(num_nodes, coords, edge_idx, node_idx, r=0.25, N=100):
#     """
#     For a single node, sample N neighbours within radius r.
#     return the indices of the neighbours
#     """
#     cluster = torch.zeros(
#         num_nodes, dtype=torch.bool)
#     source_nodes_mask = index_to_mask(
#         torch.tensor([node_idx]), num_nodes)
#     while True:
#         neighbors_mask = get_neighbors(
#             index_to_mask(source_nodes_mask, num_nodes),
#             edge_idx)
#         neighbors_mask = neighbors_mask & ~cluster
#         neighbors_idx = mask_to_index(neighbors_mask)
#         neighbors_dist = calc_dist(
#             coords, node_idx, neighbors_mask
#         )

#         neighbors_in_range = neighbors_idx[
#             neighbors_dist < r]
#         if (neighbors_in_range.shape[0] == 0):
#             break
#         else:
#             source_nodes_mask = index_to_mask(
#                 neighbors_in_range, num_nodes)
#             cluster = cluster | source_nodes_mask
#     cluster[node_idx] = False
#     return cluster


def sampler(num_nodes, coords, edge_idx, node_idx, r=0.25, N=100):
    """
    For a single node, sample N neighbours within radius r.
    return the indices of the neighbours
    """
    dist = torch.linalg.vector_norm(
        coords - coords[node_idx], dim=1
    )
    return dist < r


def get_new_edges(
        num_nodes, coords, edge_idx,
        r=0.35, M=None, dist_weight=True,
        add_nei=False,
        ):
    """
    Get the new edges for the graph.
    A useful knowledge for setting r and M:
    when on 15x15 dataset, r=0.35, M=25.
    Args:
        data: the data object
        r: the radius of the cluster
    """
    mini = 9999
    new_edges = []
    for i in range(num_nodes):
        mask = sampler(num_nodes, coords, edge_idx, i, r=r)
        cluster_idx = mask_to_index(mask)
        if (M is not None):
            # check if sampling is valid
            num_nei = len(cluster_idx)
            mini = min(mini, num_nei)
            if num_nei < M:
                raise ValueError(
                    f"The number of neighbors {num_nei} is less than M ({M})")
            if not dist_weight:
                # so the sampling
                filter_idx = torch.randperm(num_nei)[:M]
                cluster_idx = cluster_idx[filter_idx]
                # print("after sampling, ", len(cluster_idx))
            else:
                print('use dist_weight')
                dist = calc_dist(coords, i, mask)
                probs = 1 / dist
                probs = probs / probs.sum()  # normalize
                filter_idx = torch.multinomial(
                    probs, num_samples=25, replacement=False)
                cluster_idx = cluster_idx[filter_idx]
        source_idx = torch.ones(cluster_idx.shape[0], dtype=torch.long) * i
        new_edge = torch.stack([source_idx, cluster_idx], dim=0)
        new_edges.append(new_edge)
        # break
    new_edges = torch.cat(new_edges, dim=1)
    if (add_nei):
        nei_edges = torch.cat([edge_idx, new_edges], dim=1)
        return nei_edges
    else:
        return new_edges


def get_neighbors_v0(data, source_mask, edge_idx):
    """
    Get the neighbors of the source nodes
    Args:
        data: the data object, a sampler draws form the MeshDataset
        source_mask: a mask of the source nodes
        edge_idx: the edge index
    return:
        nei_mask: a mask of the neighbors
    """
    node_num = source_mask.shape[0]
    source_idxs = mask_to_index(source_mask)
    nei_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for idx in source_idxs:
        nei_nodes = edge_idx[1][edge_idx[0] == idx]
        nei_mask_i = index_to_mask(nei_nodes, node_num)
        nei_mask = nei_mask | nei_mask_i

    # substract the source nodes
    nei_mask = nei_mask & ~source_mask

    print(mask_to_index(nei_mask))
