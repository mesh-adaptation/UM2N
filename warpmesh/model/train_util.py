# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
from torch_geometric.nn import MessagePassing
__all__ = ['train', 'evaluate', 'load_model', 'TangleCounter',
           'count_dataset_tangle', 'get_jacob_det']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def jacobLoss(model, out, data, loss_func):
    jacob_det = get_jacob_det(model, data)
    u_loss = loss_func(out, data.y)
    jacob_det_loss = loss_func(
        jacob_det,
        data.jacobian_det
    )
    # print("using jacobian")
    print("u_loss: ", u_loss.item())
    print("jacob_det_loss: ", jacob_det_loss.item())
    return u_loss + 0.01 * jacob_det_loss


def get_jacob_det(model, in_data):
    in_data.x.requires_grad_(True)
    in_data.mesh_feat.requires_grad_(True)

    out = model(in_data.to(device))

    dXdx1 = torch.autograd.grad(
        outputs=out[:, 0],
        inputs=in_data.x,
        retain_graph=True,
        create_graph=False,
        grad_outputs=torch.ones_like(out[:, 0]),
    )[0][:, :2]

    dXdy1 = torch.autograd.grad(
        outputs=out[:, 1],
        inputs=in_data.x,
        retain_graph=True,
        create_graph=False,
        grad_outputs=torch.ones_like(out[:, 1]),
    )[0][:, :2]

    dXdx2 = torch.autograd.grad(
        outputs=out[:, 0],
        inputs=in_data.mesh_feat,
        retain_graph=True,
        create_graph=False,
        grad_outputs=torch.ones_like(out[:, 0]),
    )[0][:, :2]

    dXdy2 = torch.autograd.grad(
        outputs=out[:, 1],
        inputs=in_data.mesh_feat,
        retain_graph=True,
        create_graph=False,
        grad_outputs=torch.ones_like(out[:, 1]),
    )[0][:, :2]

    dXdx = dXdx1 + dXdx2
    dXdy = dXdy1 + dXdy2

    jacobian = torch.stack([dXdx, dXdy], dim=1)

    determinant = (
        jacobian[:, 0, 0] * jacobian[:, 1, 1] -
        jacobian[:, 0, 1] * jacobian[:, 1, 0]
    )
    return determinant


class TangleCounter(MessagePassing):
    """
    A PyTorch Geometric Message Passing class for counting tangles in the mesh.
    """
    def __init__(self, num_feat=10, out=16):
        super().__init__(aggr='add')
        self.num_tangle = 0

    def forward(self, x, x_new, edge_index):
        self.propagate(edge_index, x=x, x_new=x_new)
        return self.num_tangle

    def message(self, x_i, x_j, x_new_i, x_new_j):
        coord_dim = 2
        x_i = x_i[:, :coord_dim]
        x_j = x_j[:, :coord_dim]
        x_new_i = x_new_i[:, :coord_dim]
        x_new_j = x_new_j[:, :coord_dim]

        diff_x = x_i - x_j
        diff_x_new = x_new_i - x_new_j

        tangle = diff_x * diff_x_new
        tangle_arr = tangle.reshape(-1)
        self.num_tangle = torch.sum(tangle_arr < 0)
        return tangle


def count_dataset_tangle(dataset, model, device):
    """
    Computes the average number of tangles in a dataset.

    Args:
        dataset (Dataset): The PyTorch Geometric dataset.
        model (torch.nn.Module): The PyTorch model.
        device (torch.device): The device to run the computation.

    Returns:
        float: The average number of tangles in the dataset.
    """
    model.eval()
    num_tangle = 0
    for i in range(len(dataset)):
        data = dataset[i].to(device)
        with torch.no_grad():
            output_data = model(data)
            input_edge = data.edge_index
            mesh = data.x[:, :2]
            mesh_new = output_data
            Counter = TangleCounter()
            num_tangle += Counter(mesh, mesh_new, input_edge)
    num_tangle = num_tangle / len(dataset)
    return num_tangle.item()


def train(loader, model, optimizer, device, loss_func, use_jacob=False):
    """
    Trains a PyTorch model using the given data loader, optimizer,
        and loss function.

    Args:
        loader (DataLoader): DataLoader object for the training data.
        model (torch.nn.Module): The PyTorch model to train.
        optimizer (Optimizer): The optimizer (e.g., Adam, SGD).
        device (torch.device): The device to run the computation on.
        loss_func (callable): Loss function (e.g., MSE, Cross-Entropy).
        use_jacob (bool): Whether or not to use Jacobian loss.

    Returns:
        float: The average training loss across all batches.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        data = batch.to(device)
        out = model(data)
        loss = 1000*(
            loss_func(out, data.y) if not use_jacob else
            jacobLoss(model, out, data, loss_func)
        )
        if use_jacob:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return (total_loss / len(loader))


def evaluate(loader, model, device, loss_func, use_jacob=False):
    """
    Evaluates a model using the given data loader and loss function.

    Args:
        loader (DataLoader): DataLoader object for the evaluation data.
        model (torch.nn.Module): The PyTorch model to evaluate.
        device (torch.device): The device to run the computation on.
        loss_func (callable): Loss function (e.g., MSE, Cross-Entropy).
        use_jacob (bool): Whether or not to use Jacobian loss. Defaults to.

    Returns:
        float: The average evaluation loss across all batches.
    """
    model.eval()
    total_loss = 0
    for batch in loader:
        data = batch.to(device)
        loss = 0
        if use_jacob:
            out = model(data)
            loss = 1000*(
                loss_func(out, data.y) if not use_jacob else
                jacobLoss(model, out, data, loss_func)
            )
        else:
            with torch.no_grad():
                out = model(data)
                loss = 1000*(
                    loss_func(out, data.y) if not use_jacob else
                    jacobLoss(model, out, data, loss_func)
                )
        total_loss += loss.item()
    return (total_loss / len(loader))


def load_model(model, weight_path):
    """
    Loads pre-trained weights into a PyTorch model from a given file path.

    Args:
        model (torch.nn.Module): The PyTorch model.
        weight_path (str): File path to the pre-trained model weights.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    try:
        model.load_state_dict(torch.load(weight_path))
    except RuntimeError:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model
