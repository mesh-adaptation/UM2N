# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


__all__ = [
    'plot_loss', 'plot_tangle',
    'plot_mesh', 'plot_mesh_compare',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_tangle(train_tangle_arr, test_tangle_arr, epoch_arr):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.plot(epoch_arr, train_tangle_arr, label="train")
    ax.plot(epoch_arr, test_tangle_arr, label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Tangle ")
    ax.legend()
    return fig


def plot_loss(train_loss_arr, test_loss_arr, epoch_arr):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    # write loss to fig
    final_train_loss = train_loss_arr[-1]
    final_test_loss = test_loss_arr[-1]
    ax.text(0.85, 0.15, 'Final Train Loss: {:.4f}'.format(final_train_loss),
            transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.85, 0.10, 'Final Test Loss: {:.4f}'.format(final_test_loss),
            transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.plot(epoch_arr, train_loss_arr, label="train")
    ax.plot(epoch_arr, test_loss_arr, label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig


def plot_mesh(coord, face, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    vertices = [coord[face[:, i]] for i in range(face.shape[1])]
    poly_collection = PolyCollection(
        vertices, edgecolors='black', facecolors='none')
    ax.add_collection(poly_collection)
    ax.set_aspect('equal')
    return ax, fig


def plot_mesh_compare(coord_out, coord_target, face):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0], _ = plot_mesh(coord_out, face, ax=ax[0])
    ax[0].set_title("Output")
    ax[1], _ = plot_mesh(coord_target, face, ax=ax[1])
    ax[1].set_title("Target")
    return fig
