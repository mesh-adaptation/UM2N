# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.collections import PolyCollection

__all__ = [
    "plot_loss",
    "plot_tangle",
    "plot_mesh",
    "plot_mesh_compare_benchmark",
    "plot_mesh",
    "plot_mesh_compare",
    "plot_multiple_mesh_compare",
    "plot_attentions_map",
    "plot_attentions_map_compare",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    ax.text(
        0.85,
        0.15,
        "Final Train Loss: {:.4f}".format(final_train_loss),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax.text(
        0.85,
        0.10,
        "Final Test Loss: {:.4f}".format(final_test_loss),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
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
    poly_collection = PolyCollection(vertices, edgecolors="black", facecolors="none")
    ax.add_collection(poly_collection)
    ax.set_aspect("equal")
    return ax, fig


def plot_hessian(coord, face, hessian, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    vertices = [coord[face[:, i]] for i in range(face.shape[1])]
    hessian_val = [hessian[face[:, i]] for i in range(face.shape[1])]
    print(coord.shape, face.shape, max(hessian_val), min(hessian_val))
    poly_collection = PolyCollection(
        vertices, edgecolors="black", facecolors=hessian_val
    )
    ax.add_collection(poly_collection)
    ax.set_aspect("equal")
    return ax, fig


def plot_attention(attentions, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(attentions[-1])
    ax.set_aspect("equal")
    return ax, fig


def plot_mesh_compare(coord_out, coord_target, face):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0], _ = plot_mesh(coord_out, face, ax=ax[0])
    ax[0].set_title("Output")
    ax[1], _ = plot_mesh(coord_target, face, ax=ax[1])
    ax[1].set_title("Target")
    return fig


def plot_mesh_compare_benchmark(
    coord_out,
    coord_target,
    face,
    deform_loss,
    pde_loss_model,
    pde_loss_reduction_model,
    pde_loss_MA,
    pde_loss_reduction_MA,
    tangle,
):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0], _ = plot_mesh(coord_out, face, ax=ax[0])
    ax[0].set_title(
        rf"Output | Deform Loss: {deform_loss:.2f} | PDE Loss (model): {pde_loss_model:.5f} ({pde_loss_reduction_model*100:.2f}$\%$) |Tangle: {tangle:.2f}"
    )
    ax[1], _ = plot_mesh(coord_target, face, ax=ax[1])
    ax[1].set_title(
        rf"Target | PDE Loss (MA): {pde_loss_MA:.5f} ({pde_loss_reduction_MA*100:.2f}$\%$)"
    )
    return fig


def plot_multiple_mesh_compare(
    out_mesh_collections, out_loss_collections, target_mesh, target_face
):
    model_names = list(out_mesh_collections.keys())
    num_models = len(model_names)
    num_samples = len(out_mesh_collections[model_names[0]])
    fig, ax = plt.subplots(
        num_samples, num_models + 1, figsize=(4 * num_models + 1, 4 * num_samples)
    )

    for n_model, model_name in enumerate(model_names):
        all_mesh = out_mesh_collections[model_name]
        all_loss = out_loss_collections[model_name]
        for n_sample in range(num_samples):
            ax[n_sample, n_model], _ = plot_mesh(
                all_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, n_model]
            )
            ax[n_sample, n_model].set_title(
                f"{model_name} (loss: {all_loss[n_sample]:.2f})", fontsize=12
            )
    # Plot the ground truth
    for n_sample in range(num_samples):
        ax[n_sample, num_models], _ = plot_mesh(
            target_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, num_models]
        )
        ax[n_sample, num_models].set_title("Target", fontsize=16)
    return fig


def plot_attentions_map(out_atten_collections, out_loss_collections):
    model_names = list(out_atten_collections.keys())
    num_models = len(model_names)
    num_samples = len(out_atten_collections[model_names[0]])
    fig, ax = plt.subplots(
        num_samples, num_models, figsize=(4 * num_models, 4 * num_samples)
    )

    for n_model, model_name in enumerate(model_names):
        all_atten = out_atten_collections[model_name]
        all_loss = out_loss_collections[model_name]
        for n_sample in range(num_samples):
            ax[n_sample, n_model], _ = plot_attention(
                all_atten[n_sample], ax=ax[n_sample, n_model]
            )
            ax[n_sample, n_model].set_title(
                f"{model_name} (loss: {all_loss[n_sample]:.2f})", fontsize=16
            )
    # # Plot the ground truth
    # for n_sample in range(num_samples):
    #     ax[n_sample, num_models], _ = plot_mesh(target_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, num_models])
    #     ax[n_sample, num_models].set_title("Target", fontsize=16)
    return fig


def plot_attention_on_mesh(coord, face, atten_weights, selected_node=200, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    # vertices = [coord[face[:, i]] for i in range(face.shape[1])]
    vertices = coord
    # print(vertices)
    vertices_index = [x for x in range(len(vertices))]
    vertices_pos = {}
    for node in vertices_index:
        vertices_pos[node] = vertices[node]

    G = nx.DiGraph()
    # Add nodes
    for node in vertices_index:
        G.add_node(node)

    # Add edges with attention weights
    # for i in range(1):
    node_color_list = [0.0 for _ in range(len(vertices_index))]
    i = selected_node
    for j in range(len(vertices_index)):
        if i != j:  # Assuming no self-loops
            if len(atten_weights.shape) == 2:
                G.add_edge(
                    vertices_index[i], vertices_index[j], weight=atten_weights[i][j]
                )
                node_color_list[j] = atten_weights[i][j]
            else:
                G.add_edge(
                    vertices_index[i], vertices_index[j], weight=atten_weights[j]
                )
                node_color_list[j] = atten_weights[j]

    print(f"num nodes: {len(vertices_index)}, node color list: {len(node_color_list)}")
    # Draw the graph
    # pos = nx.spring_layout(G)  # You can try different layouts
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]

    # nx.draw(G, vertices_pos, ax=ax, with_labels=False, edge_color=weights, node_size=0.1, width=0.05, arrowsize=0.02, edge_cmap=mpl.colormaps['plasma'])

    # Draw the node of intetest
    nx.draw(
        G,
        vertices_pos,
        nodelist=[list(G)[selected_node]],
        ax=ax,
        arrows=None,
        with_labels=False,
        node_color="black",
        node_size=5.0,
        width=0.0,
        arrowsize=0.0,
    )
    # Draw the attention on other nodes
    nx.draw(
        G,
        vertices_pos,
        ax=ax,
        arrows=None,
        with_labels=False,
        node_color=node_color_list,
        node_size=1.0,
        width=0.0,
        arrowsize=0.0,
        cmap=mpl.colormaps["plasma"],
    )

    # Create colorbar for the edges
    sm = plt.cm.ScalarMappable(
        cmap=mpl.colormaps["plasma"],
        norm=plt.Normalize(vmin=min(weights), vmax=max(weights)),
    )
    sm.set_array([])

    # cax = plt.axes([0.95, 0.05, 0.05,0.9 ])
    plt.colorbar(
        sm, ax=ax, orientation="vertical", shrink=0.4, label="Attention weights"
    )
    plt.tight_layout()

    ax.set_aspect("equal")
    return ax, fig


def plot_attentions_map_compare(
    out_mesh_collections,
    out_loss_collections,
    out_atten_collections,
    target_hessian,
    target_mesh,
    target_face,
    selected_node=200,
):
    model_names = list(out_mesh_collections.keys())
    num_models = len(model_names)
    num_samples = len(out_mesh_collections[model_names[0]])
    fig, ax = plt.subplots(
        num_samples, num_models + 1, figsize=(4 * num_models + 1, 4 * num_samples)
    )
    for n_model, model_name in enumerate(model_names):
        all_atten = out_atten_collections[model_name]
        all_mesh = out_mesh_collections[model_name]
        all_loss = out_loss_collections[model_name]
        for n_sample in range(num_samples):
            ax[n_sample, n_model], _ = plot_attention_on_mesh(
                all_mesh[n_sample],
                target_face[n_sample],
                atten_weights=all_atten[n_sample][-1],
                selected_node=selected_node,
                ax=ax[n_sample, n_model],
            )
            ax[n_sample, n_model].set_title(
                f"{model_name} (loss: {all_loss[n_sample]:.2f})", fontsize=16
            )
    # Plot the ground truth
    for n_sample in range(num_samples):
        # ax[n_sample, num_models], _ = plot_mesh(target_mesh[n_sample], target_face[n_sample], ax=ax[n_sample, num_models])
        ax[n_sample, num_models], _ = plot_attention_on_mesh(
            target_mesh[n_sample],
            target_face[n_sample],
            target_hessian[n_sample],
            selected_node=selected_node,
            ax=ax[n_sample, num_models],
        )
        ax[n_sample, num_models].set_title("Target", fontsize=16)
    return fig
