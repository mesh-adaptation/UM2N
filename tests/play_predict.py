import os
import warpmesh as wm
import torch
import firedrake as fd
import movement as mv
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


problem = "helmholtz"

model = wm.MRN(gfe_in_c=2, lfe_in_c=4, num_loop=10)

weight_path = "/Users/cw1722/Downloads/model_1099.pth"
weight_decay = 5e-4
train_batch_size = 20

pre_train_weight_path = None

simple_u = True
data_type = "smpl" if simple_u else "cmplx"

n_dist = None
max_dist = 6
n_elem_x = 20
n_elem_y = 20
num_samples = 400

num_epochs = 50
print_interval = 1
save_interval = 1

loss_func = torch.nn.L1Loss()
# loss_func = torch.nn.MSELoss()

normalise = True
# normalise = False

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
    "conv_hessian_norm",
]

project_dir = os.path.dirname(
    os.path.dirname((os.path.abspath(__file__))))
data_set_path = os.path.join(project_dir, "data/dataset/helmholtz")
data_path = os.path.join(
    data_set_path,
    "z=<0,1>_ndist={}_max_dist={}_<{}x{}>_n={}_{}/"
    .format(n_dist, max_dist, n_elem_x, n_elem_y, num_samples, data_type)
)


def plot_prediction(data_set, model, prediction_dir, mode):
    num_data = len(data_set)
    for idx in range(num_data):
        val_item = data_set[idx]
        out = model(val_item.to(device))
        # calculate the loss
        loss = 1000*torch.functional.F.mse_loss(
            out, val_item.y).item()
        out = out.detach().numpy()
        # construct the mesh
        val_mesh = fd.UnitSquareMesh(n_elem_x, n_elem_y)
        val_new_mesh = fd.UnitSquareMesh(n_elem_x, n_elem_y)
        # init checker
        checker = mv.MeshTanglingChecker(val_new_mesh, mode='warn')
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
        ax2.text(0.5, -0.05, f"Num Tangle: {num_tangle}",
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=14)
        fig.text(0.5, 0.01, f'Loss: {loss:.4f}',
                 ha='center', va='center', fontsize=16)
        fig.savefig(
            os.path.join(
                prediction_dir, f"{mode}_plot_{idx}.png")
        )


prediction_dir = "/Users/cw1722/Documents/irp/irp-cw1722/data/temp"

train_set = wm.MeshDataset(
    os.path.join(data_path, "train"),
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)
test_set = wm.MeshDataset(
    os.path.join(data_path, "test"),
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)
val_set = wm.MeshDataset(
    os.path.join(data_path, "val"),
    transform=wm.normalise if normalise else None,
    x_feature=x_feat,
    mesh_feature=mesh_feat,
    conv_feature=conv_feat,
)


model = model.to(device)


if __name__ == "__main__":
    state_dict = torch.load(
        weight_path,
        map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    plot_prediction(test_set, model, prediction_dir, mode="test")
    plot_prediction(val_set, model, prediction_dir, mode="val")
