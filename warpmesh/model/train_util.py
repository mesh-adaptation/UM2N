# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
__all__ = ['train', 'evaluate', 'load_model', 'TangleCounter',
           'count_dataset_tangle', 'get_jacob_det']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_face_area(coord, face):
    """
    Calculates the area of a face. using formula:
        area = 0.5 * (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2))
    Args:
        coord (torch.Tensor): The coordinates.
        face (torch.Tensor): The face tensor.
    """
    x = coord[:, 0][face]
    y = coord[:, 1][face]

    area = 0.5 * (
        x[0, :] * (y[1, :] - y[2, :]) +
        x[1, :] * (y[2, :] - y[0, :]) +
        x[2, :] * (y[0, :] - y[1, :])
    )
    return area


def get_inversion_loss(out_coord, in_coord, face, batch_size, scaler=100):
    """
    Calculates the inversion loss for a batch of meshes.
    Args:
        out_coord (torch.Tensor): The output coordinates.
        in_coord (torch.Tensor): The input coordinates.
        face (torch.Tensor): The face tensor.
        batch_size (int): The batch size.
        alpha (float): The loss weight.
    """
    out_area = get_face_area(out_coord, face)
    in_area = get_face_area(in_coord, face)
    # restore the sign of the area, ans scale it
    out_area = scaler * torch.sign(in_area) * out_area
    # mask for negative area
    neg_mask = out_area < 0
    neg_area = out_area[neg_mask]
    # loss should be positive, so we are using -1 here.
    loss = (-1 * (neg_area.sum()) / batch_size)
    return loss


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
    This class is deprecated, do not use this option unless you know what you
    are doing.
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


def count_dataset_tangle(dataset, model, device, method="inversion"):
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
    if (method == "inversion"):
        loader = DataLoader(dataset=dataset, batch_size=10,
                            shuffle=False)
        for data in loader:
            output_data = model(data.to(device))
            out_area = get_face_area(output_data, data.face)
            in_area = get_face_area(data.x[:, :2], data.face)
            # restore the sign of the area
            out_area = torch.sign(in_area) * out_area
            # mask for negative area
            neg_mask = out_area < 0
            neg_area = out_area[neg_mask]
            # calculate the loss, we want it normalized by the batch size
            # and loss should be positive, so we are using -1 here.
            num_tangle += len(neg_area)
        return num_tangle / len(dataset)

    # deprecated, do not use this option unless you know what you are doing
    elif (method == "msg"):
        for i in range(len(dataset)):
            data = dataset[i].to(device)
            with torch.no_grad():
                output_data = model(data)
                input_edge = data.edge_index
                mesh = data.x[:, :2]
                mesh_new = output_data
                Counter = TangleCounter()
                num_tangle += Counter(mesh, mesh_new, input_edge).item()
        num_tangle = num_tangle / len(dataset)
        return num_tangle


def train(
        loader, model, optimizer, device, loss_func,
        use_jacob=False, use_inversion_loss=False, scaler=100):
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
    bs = loader.batch_size
    model.train()
    total_loss = 0
    total_deform_loss = 0
    total_inversion_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        data = batch.to(device)
        out = model(data)
        loss = 0
        inversion_loss = 0
        deform_loss = 0
        # deformation loss
        deform_loss = 1000*(
            loss_func(out, data.y) if not use_jacob else
            jacobLoss(model, out, data, loss_func)
        )
        # Inversion loss
        if use_inversion_loss:
            inversion_loss = get_inversion_loss(
                out, data.x[:, :2], data.face, bs, scaler)
        loss = inversion_loss + deform_loss
        # Jacobian loss
        if use_jacob:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        optimizer.step()
        total_loss += loss.item()
        total_deform_loss += deform_loss.item()
        total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0 # noqa
    if (use_inversion_loss):
        return {
            "total_loss": total_loss / len(loader),
            "deform_loss": total_deform_loss / len(loader),
            "inversion_loss": total_inversion_loss / len(loader)
        }
    return (total_loss / len(loader))


def evaluate(
        loader, model, device, loss_func, use_jacob=False,
        use_inversion_loss=False, scaler=100):
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
    bs = loader.batch_size
    model.eval()
    total_loss = 0
    total_deform_loss = 0
    total_inversion_loss = 0
    for batch in loader:
        data = batch.to(device)
        loss = 0
        deform_loss = 0
        inversion_loss = 0
        if use_jacob:
            out = model(data)
            loss = 1000*(
                loss_func(out, data.y) if not use_jacob else
                jacobLoss(model, out, data, loss_func)
            )
        else:
            with torch.no_grad():
                out = model(data)
                deform_loss = 1000*(
                    loss_func(out, data.y) if not use_jacob else
                    jacobLoss(model, out, data, loss_func)
                )
                inversion_loss = 0
                if use_inversion_loss:
                    inversion_loss = get_inversion_loss(
                        out, data.x[:, :2], data.face, bs, scaler)
                loss = inversion_loss + deform_loss
                total_loss += loss.item()
                total_deform_loss += deform_loss.item()
                total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0  # noqa
    if (use_inversion_loss):
        return {
            "total_loss": total_loss / len(loader),
            "deform_loss": total_deform_loss / len(loader),
            "inversion_loss": total_inversion_loss / len(loader)
        }
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
