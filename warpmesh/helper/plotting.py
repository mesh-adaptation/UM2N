# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

__all__ = [
    'plot_loss', 'plot_tangle',
    'plot_mesh',
    'plot_mesh_compare_benchmark',
    'plot_mesh', 'plot_mesh_compare', 'plot_multiple_mesh_compare',
    'plot_attentions_map'
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

def plot_attention(attentions, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(attentions[-1])
    ax.set_aspect('equal')
    return ax, fig

def plot_mesh_compare(coord_out, coord_target, face):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0], _ = plot_mesh(coord_out, face, ax=ax[0])
    ax[0].set_title("Output")
    ax[1], _ = plot_mesh(coord_target, face, ax=ax[1])
    ax[1].set_title("Target")
    return fig


def plot_mesh_compare_benchmark(coord_out, coord_target, face, loss, tangle):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0], _ = plot_mesh(coord_out, face, ax=ax[0])
    ax[0].set_title(f"Output | Loss: {loss:.2f} | Tangle: {tangle:.2f}")
    ax[1], _ = plot_mesh(coord_target, face, ax=ax[1])
    ax[1].set_title("Target")
    return fig

def plot_multiple_mesh_compare(out_mesh_collections, out_loss_collections, target_mesh, target_face):
    model_names = list(out_mesh_collections.keys())
    num_models = len(model_names)
    num_samples = len(out_mesh_collections[model_names[0]])
    fig, ax = plt.subplots(num_samples, num_models+1, figsize=(4*num_models + 1, 4*num_samples))
    
    for n_model, model_name in enumerate(model_names):
        all_mesh = out_mesh_collections[model_name]
        all_loss = out_loss_collections[model_name]
        for n_sample in range(num_samples):
            ax[n_sample, n_model], _ = plot_mesh(all_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, n_model])
            ax[n_sample, n_model].set_title(f"{model_name} (loss: {all_loss[n_sample]:.2f})", fontsize=16)
    # Plot the ground truth
    for n_sample in range(num_samples):
        ax[n_sample, num_models], _ = plot_mesh(target_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, num_models])
        ax[n_sample, num_models].set_title("Target", fontsize=16)
    return fig


def plot_attentions_map(out_atten_collections, out_loss_collections):
    model_names = list(out_atten_collections.keys())
    num_models = len(model_names)
    num_samples = len(out_atten_collections[model_names[0]])
    fig, ax = plt.subplots(num_samples, num_models, figsize=(4*num_models, 4*num_samples))
    
    for n_model, model_name in enumerate(model_names):
        all_atten = out_atten_collections[model_name]
        all_loss = out_loss_collections[model_name]
        for n_sample in range(num_samples):
            ax[n_sample, n_model], _ = plot_attention(all_atten[n_sample], ax=ax[n_sample, n_model])
            ax[n_sample, n_model].set_title(f"{model_name} (loss: {all_loss[n_sample]:.2f})", fontsize=16)
    # # Plot the ground truth
    # for n_sample in range(num_samples):
    #     ax[n_sample, num_models], _ = plot_mesh(target_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, num_models])
    #     ax[n_sample, num_models].set_title("Target", fontsize=16)
    return fig