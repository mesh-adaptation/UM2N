# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import os

import firedrake as fd
import matplotlib.pyplot as plt
import movement as mv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["check_dataset_tangle", "plot_prediction", "plot_sample"]


def check_tangle_pi(model, x):
    pass


def check_dataset_tangle(
    dataset,
    model,
    n_elem_x,
    n_elem_y,
):
    """
    Return the percentage of tangling grid of a mesh in a dataset.
    """
    num_tangled = 0
    for idx in range(len(dataset)):
        mesh = fd.UnitSquareMesh(n_elem_x, n_elem_y)
        checker = mv.MeshTanglingChecker(mesh, mode="warn")
        check_item = dataset[idx]
        out = model(check_item.to(device)).detach().numpy()
        mesh.coordinates.dat.data[:, 0] = out[:, 0]
        mesh.coordinates.dat.data[:, 1] = out[:, 1]
        num_tangled += checker.check()
    return num_tangled / len(dataset)


def plot_prediction(
    data_set, model, prediction_dir, mode, n_elem_x, n_elem_y, loss_fn, savefig=True
):
    num_data = len(data_set)
    for idx in range(num_data):
        val_item = data_set[idx]
        plot_sample(
            model,
            val_item,
            prediction_dir,
            loss_fn,
            n_elem_x,
            n_elem_y,
            idx,
            mode,
            savefig,
        )


def plot_sample(
    model,
    val_item,
    prediction_dir,
    loss_fn,
    n_elem_x,
    n_elem_y,
    idx,
    mode,
    savefig=True,
):
    out = model(val_item.to(device))
    # calculate the loss
    loss = 1000 * loss_fn(out, val_item.y).item()
    out = out.detach().numpy()
    # construct the mesh
    val_mesh = fd.UnitSquareMesh(n_elem_x, n_elem_y)
    val_new_mesh = fd.UnitSquareMesh(n_elem_x, n_elem_y)
    # init checker
    checker = mv.MeshTanglingChecker(val_new_mesh, mode="warn")
    # construct the predicted/target mesh
    val_mesh.coordinates.dat.data[:] = val_item.y[:]
    val_new_mesh.coordinates.dat.data[:] = out[:]
    num_tangle = checker.check()
    # plot the mesh, tangle/loss info
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
    fd.triplot(val_mesh, axes=ax1)
    fd.triplot(val_new_mesh, axes=ax2)
    ax1.set_title("Target mesh")
    ax2.set_title("Predicted mesh")
    ax2.text(
        0.5,
        -0.05,
        f"Num Tangle: {num_tangle}",
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=14,
    )
    fig.text(0.5, 0.01, f"Loss: {loss:.4f}", ha="center", va="center", fontsize=16)
    if savefig:
        fig.savefig(os.path.join(prediction_dir, f"{mode}_plot_{idx}.png"))
