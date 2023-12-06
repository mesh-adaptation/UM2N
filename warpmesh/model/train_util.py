# Author: Chunyang Wang
# GitHub Username: acse-cw1722

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

__all__ = ['train', 'evaluate', 'load_model', 'TangleCounter',
           'count_dataset_tangle', 'get_jacob_det',
           'get_inversion_diff_loss', 'get_face_area',
           'count_dataset_tangle', 'get_jacob_det', 'get_face_area',
           'get_inversion_loss', 'get_inversion_node_loss',
           'get_area_loss', 'evaluate_repeat_sampling',
           'count_dataset_tangle_repeat_sampling',
           'evaluate_repeat', 'get_sample_tangle'
           ]

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


def get_inversion_loss(out_coord, in_coord, face,
                       batch_size,
                       scheme="relu",
                       scaler=100):
    """
    Calculates the inversion loss for a batch of meshes.
    Args:
        out_coord (torch.Tensor): The output coordinates.
        in_coord (torch.Tensor): The input coordinates.
        face (torch.Tensor): The face tensor.
        batch_size (int): The batch size.
        alpha (float): The loss weight.
    """
    loss = None
    out_area = get_face_area(out_coord, face)
    in_area = get_face_area(in_coord, face)
    # restore the sign of the area, ans scale it
    out_area = torch.sign(in_area) * out_area
    # hard penalty, use hard condition to penalize the negative area
    if (scheme == "hard"):
        # mask for negative area
        neg_mask = out_area < 0
        neg_area = out_area[neg_mask]
        tar_area = in_area[neg_mask]
        # loss should be positive, so we are using -1 here.
        loss = (-1 * ((neg_area / torch.abs(tar_area)).sum()) / batch_size)
    # soft penalty, peanlize the negative area harder than the positive area
    elif (scheme == "relu"):
        loss = (torch.nn.ReLU()(
            -1 * (out_area / torch.abs(in_area))
            ).sum() / batch_size)
    elif (scheme == "log"):
        epsilon = 1e-8
        loss = (
            -1 * torch.log(
                -1 * (out_area / torch.abs(in_area))).sum() + epsilon
            ) / batch_size
    return scaler * loss


def get_inversion_diff_loss(out_coord, tar_coord, face,
                            batch_size, scaler=100):
    """
    Calculates the inversion difference loss for a batch of meshes.
    That is the difference between the output area and the input area,
    in terms of the invereted elements.
    Args:
        out_coord (torch.Tensor): The output coordinates.
        tar_coord (torch.Tensor): The target coordinates.
        face (torch.Tensor): The face tensor.
        batch_size (int): The batch size.
        alpha (float): The loss weight.
    """
    out_area = get_face_area(out_coord, face)
    tar_area = get_face_area(tar_coord, face)
    # restore the sign of the area, ans scale it
    out_area = scaler * torch.sign(tar_area) * out_area
    tar_area = scaler * torch.sign(tar_area) * tar_area
    # mask for negative area
    neg_mask = out_area < 0
    inversion_diff = (
        tar_area[neg_mask] - out_area[neg_mask]
    )
    # loss should be positive, so we are using -1 here.
    loss = (inversion_diff.sum() / batch_size)
    return loss


def get_inversion_node_loss(out_coord, tar_coord, face, batch_size,
                            scaler=1000):
    """
    Calculates the loss between the ouput node and input node, for the inverted
    elements. This will penalise the node which are involved in the tangled
    elements.
    Args:
        out_coord (torch.Tensor): The output coordinates.
        tar_coord (torch.Tensor): The target coordinates.
        face (torch.Tensor): The face tensor.
        batch_size (int): The batch size.
        alpha (float): The loss weight.
    """
    loss = torch.nn.L1Loss()
    out_area = get_face_area(out_coord, face)
    tar_area = get_face_area(tar_coord, face)
    # restore the sign of the area, ans scale it
    out_area = torch.sign(tar_area) * out_area
    tar_area = torch.sign(tar_area) * tar_area
    # mask for negative area
    neg_mask = out_area < 0
    neg_face = face[:, neg_mask]
    neg_face = neg_face.reshape(-1)
    inv_nodes = out_coord[neg_face]
    tar_nodes = tar_coord[neg_face]

    node_diff = scaler * loss(inv_nodes, tar_nodes)

    loss = (node_diff / batch_size)
    zero = torch.tensor(
        0.0, device=out_coord.device,
        dtype=out_coord.dtype,
        requires_grad=True)

    return loss if len(inv_nodes) > 0 else zero


def get_area_loss(out_coord, tar_coord, face, batch_size, scaler=100):
    out_area = get_face_area(out_coord, face)
    tar_area = get_face_area(tar_coord, face)
    # restore the sign of the area, ans scale it
    out_area = scaler * torch.sign(tar_area) * out_area
    tar_area = scaler * torch.sign(tar_area) * tar_area
    # mask for negative area
    area_diff = torch.abs(
        tar_area - out_area
    )
    # area_diff = tar_area - out_area + 100
    # loss should be positive, so we are using -1 here.
    loss = ((area_diff.sum()) / batch_size)
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


def train(
        loader, model, optimizer, device, loss_func,
        use_jacob=False,
        use_inversion_loss=False,
        use_inversion_diff_loss=False,
        use_area_loss=False,
        scaler=100):
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
    total_inversion_diff_loss = 0
    total_area_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        data = batch.to(device)
        out = model(data)
        loss = 0
        inversion_loss = 0
        deform_loss = 0
        inversion_diff_loss = 0
        area_loss = 0
        # deformation loss
        deform_loss = 1000*(
            loss_func(out, data.y) if not use_jacob else
            jacobLoss(model, out, data, loss_func)
        )
        # Inversion loss
        if use_inversion_loss:
            inversion_loss = get_inversion_loss(
                out, data.y, data.face,
                batch_size=bs, scaler=scaler)
        if use_area_loss:
            area_loss = get_area_loss(
                out, data.y, data.face, bs, scaler)

        loss = (
            deform_loss +
            inversion_loss +
            inversion_diff_loss +
            area_loss
        )
        # Jacobian loss
        if use_jacob:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        optimizer.step()
        total_loss += loss.item()
        total_deform_loss += deform_loss.item()
        total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0 # noqa
        total_inversion_diff_loss += inversion_diff_loss.item() if use_inversion_diff_loss else 0 # noqa
        total_area_loss += area_loss.item() if use_area_loss else 0

    res = {
        "total_loss": total_loss / len(loader),
        "deform_loss": total_deform_loss / len(loader),
    }
    if (use_inversion_loss):
        res["inversion_loss"] = total_inversion_loss / len(loader)
    if (use_inversion_diff_loss):
        res["inversion_diff_loss"] = total_inversion_diff_loss / len(loader)
    if (use_area_loss):
        res["area_loss"] = total_area_loss / len(loader)

    return res


def evaluate(
        loader, model, device, loss_func,
        use_jacob=False,
        use_inversion_loss=False,
        use_inversion_diff_loss=False,
        use_area_loss=False,
        scaler=100):
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
    total_inversion_diff_loss = 0
    total_area_loss = 0
    for batch in loader:
        data = batch.to(device)
        loss = 0
        deform_loss = 0
        inversion_loss = 0
        inversion_diff_loss = 0
        area_loss = 0

        with torch.no_grad():
            out = model(data)
            deform_loss = 1000*(
                loss_func(out, data.y) if not use_jacob else
                jacobLoss(model, out, data, loss_func)
            )
            inversion_loss = 0
            if use_inversion_loss:
                inversion_loss = get_inversion_loss(
                    out, data.y, data.face,
                    batch_size=bs, scaler=scaler)
            if use_area_loss:
                area_loss = get_area_loss(
                    out, data.y, data.face, bs, scaler)

            loss = inversion_loss + deform_loss
            total_loss += loss.item()
            total_deform_loss += deform_loss.item()
            total_inversion_diff_loss += inversion_diff_loss.item() if use_inversion_diff_loss else 0 # noqa
            total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0  # noqa
            total_area_loss += area_loss.item() if use_area_loss else 0
    res = {
        "total_loss": total_loss / len(loader),
        "deform_loss": total_deform_loss / len(loader),
    }
    if (use_inversion_loss):
        res["inversion_loss"] = total_inversion_loss / len(loader)
    if (use_inversion_diff_loss):
        res["inversion_diff_loss"] = total_inversion_diff_loss / len(loader)
    if (use_area_loss):
        res["area_loss"] = total_area_loss / len(loader)
    return res


def get_sample_tangle(out_coords, in_coords, face):
    """
    Return the number of tangled elements in a single sample.
    """
    out_area = get_face_area(out_coords, face)
    in_area = get_face_area(in_coords, face)
    out_area = torch.sign(in_area) * out_area
    neg_mask = out_area < 0
    neg_area = out_area[neg_mask]
    num_tangle = len(neg_area)
    return num_tangle


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
        loader = DataLoader(dataset=dataset, batch_size=1,
                            shuffle=False)
        for data in loader:
            with torch.no_grad():
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


def evaluate_repeat_sampling(
        dataset, model, device, loss_func,
        use_inversion_loss=False,
        use_inversion_diff_loss=False,
        use_area_loss=False,
        scaler=100,
        batch_size=5,
        num_samples=1
        ):
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
    bs = batch_size
    loaders = [
        DataLoader(dataset=dataset, batch_size=bs, shuffle=False)
        for i in range(num_samples)
    ]
    data_iters = [
        iter(loader) for loader in loaders
    ]
    total_loss = 0
    total_deform_loss = 0
    total_inversion_loss = 0
    total_inversion_diff_loss = 0
    total_area_loss = 0
    for i in range(len(loaders[0])):
        data_list = [next(data_iter) for data_iter in data_iters]
        data_list = [data.to(device) for data in data_list]
        loss = 0
        deform_loss = 0
        inversion_loss = 0
        inversion_diff_loss = 0
        area_loss = 0

        with torch.no_grad():
            out = [model(data) for data in data_list]
            out = torch.stack(out, dim=0)
            out = torch.mean(out, dim=0)
            deform_loss = 1000*(
                loss_func(out, data_list[0].y)
            )
            if use_area_loss:
                area_loss = get_area_loss(
                    out, data_list[0].y, data_list[0].face, bs, scaler)

            loss = inversion_loss + deform_loss
            total_loss += loss.item()
            total_deform_loss += deform_loss.item()
            total_inversion_diff_loss += inversion_diff_loss.item() if use_inversion_diff_loss else 0 # noqa
            total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0  # noqa
            total_area_loss += area_loss.item() if use_area_loss else 0
    res = {
        "total_loss": total_loss / len(loaders[0]),
        "deform_loss": total_deform_loss / len(loaders[0]),
    }
    if (use_inversion_loss):
        res["inversion_loss"] = total_inversion_loss / len(loaders[0])
    if (use_inversion_diff_loss):
        res["inversion_diff_loss"] = total_inversion_diff_loss / len(loaders[0])  # noqa
    if (use_area_loss):
        res["area_loss"] = total_area_loss / len(loaders[0])
    return res


def count_dataset_tangle_repeat_sampling(
        dataset, model, device, num_samples=1):
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
    loaders = [
        DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        for i in range(num_samples)
    ]
    data_iters = [
        iter(loader) for loader in loaders
    ]
    for i in range(len(loaders[0])):
        with torch.no_grad():
            data_list = [next(data_iter) for data_iter in data_iters]
            data_list = [data.to(device) for data in data_list]
            output_data = [model(data.to(device)) for data in data_list]
            output_data = torch.stack(output_data, dim=0)
            output_data = torch.mean(output_data, dim=0)
            out_area = get_face_area(output_data, data_list[0].face)
            in_area = get_face_area(data_list[0].x[:, :2], data_list[0].face)
            # restore the sign of the area
            out_area = torch.sign(in_area) * out_area
            # mask for negative area
            neg_mask = out_area < 0
            neg_area = out_area[neg_mask]
            # calculate the loss, we want it normalized by the batch size
            # and loss should be positive, so we are using -1 here.
            num_tangle += len(neg_area)
    return num_tangle / len(dataset)


def evaluate_repeat(
        dataset, model, device, loss_func,
        scaler=100,
        num_repeat=1
        ):
    """
    Evaluates model performance when sampling for different number of times.
    this function will evaluate:
        1. the average loss
        2. the average number of tangles

    Args:
        dataset (MeshDataset): The target dataset to evaluate.
        model (torch.nn.Module): The PyTorch model to evaluate.
        device (torch.device): The device to run the computation on.
        loss_func (callable): Loss function (e.g., MSE, Cross-Entropy).

    Returns:
        float: The average evaluation loss across all batches.
    """
    model.eval()
    loaders = [
        DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        for i in range(num_repeat)
    ]
    data_iters = [iter(loader) for loader in loaders]

    num_tangle = 0
    total_loss = 0
    total_deform_loss = 0
    total_area_loss = 0

    for i in range(len(loaders[0])):
        data_list = [next(data_iter) for data_iter in data_iters]
        data_list = [data.to(device) for data in data_list]

        loss = 0
        deform_loss = 0
        inversion_loss = 0
        area_loss = 0

        with torch.no_grad():
            out = [model(data) for data in data_list]
            out = torch.stack(out, dim=0)
            out = torch.mean(out, dim=0)
            # calculate the loss
            deform_loss = 1000*(
                loss_func(out, data_list[0].y)
            )
            area_loss = get_area_loss(
                out, data_list[0].y, data_list[0].face, 1, scaler)

            loss = inversion_loss + deform_loss
            total_loss += loss.item()
            total_deform_loss += deform_loss.item()
            total_area_loss += area_loss.item()

            # calculate the number of tangles
            in_area = get_face_area(data_list[0].x[:, :2], data_list[0].face)
            out_area = get_face_area(out, data_list[0].face)
            out_area = torch.sign(in_area) * out_area
            neg_mask = out_area < 0
            neg_area = out_area[neg_mask]
            num_tangle += len(neg_area)

    return {
        "total_loss": total_loss / len(dataset),
        "deform_loss": total_deform_loss / len(dataset),
        "tangle": num_tangle / len(dataset),
        "area_loss": total_area_loss / len(dataset)
    }


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
