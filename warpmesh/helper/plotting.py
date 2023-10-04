# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
import matplotlib.pyplot as plt

__all__ = ['plot_loss', 'plot_tangle']

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
