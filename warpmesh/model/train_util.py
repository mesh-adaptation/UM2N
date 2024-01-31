# Author: Chunyang Wang
# GitHub Username: acse-cw1722
# Modified by Mingrui Zhang

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph

__all__ = ['train', 'train_unsupervised', 'evaluate', 'evaluate_unsupervised', 'load_model', 'TangleCounter',
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



# def count_dataset_tangle(dataset, model, device, method="inversion"):
#     """
#     Computes the average number of tangles in a dataset.

#     Args:
#         dataset (Dataset): The PyTorch Geometric dataset.
#         model (torch.nn.Module): The PyTorch model.
#         device (torch.device): The device to run the computation.

#     Returns:
#         float: The average number of tangles in the dataset.
#     """
#     model.eval()
#     num_tangle = 0
#     if (method == "inversion"):
#         loader = DataLoader(dataset=dataset, batch_size=1,
#                             shuffle=False)
#         for data in loader:
#             with torch.no_grad():
#                 output_data = model(data.to(device))
#                 out_area = get_face_area(output_data, data.face)
#                 in_area = get_face_area(data.x[:, :2], data.face)
#                 # restore the sign of the area
#                 out_area = torch.sign(in_area) * out_area
#                 # mask for negative area
#                 neg_mask = out_area < 0
#                 neg_area = out_area[neg_mask]
#                 # calculate the loss, we want it normalized by the batch size
#                 # and loss should be positive, so we are using -1 here.
#                 num_tangle += len(neg_area)
#         return num_tangle / len(dataset)

#     # deprecated, do not use this option unless you know what you are doing
#     elif (method == "msg"):
#         for i in range(len(dataset)):
#             data = dataset[i].to(device)
#             with torch.no_grad():
#                 output_data = model(data)
#                 input_edge = data.edge_index
#                 mesh = data.x[:, :2]
#                 mesh_new = output_data
#                 Counter = TangleCounter()
#                 num_tangle += Counter(mesh, mesh_new, input_edge).item()
#         num_tangle = num_tangle / len(dataset)
#         return num_tangle



# def count_dataset_tangle(dataset, model, device, method="inversion"):
#     """
#     Computes the average number of tangles in a dataset.

#     Args:
#         dataset (Dataset): The PyTorch Geometric dataset.
#         model (torch.nn.Module): The PyTorch model.
#         device (torch.device): The device to run the computation.

#     Returns:
#         float: The average number of tangles in the dataset.
#     """
#     model.eval()
#     num_tangle = 0
#     if (method == "inversion"):
#         loader = DataLoader(dataset=dataset, batch_size=1,
#                             shuffle=False)
#         for data in loader:

#             (output_coord, model_raw_output, out_monitor), (phix, phiy) = model(data.to(device))

#             # Compute the new mesh coord given model output phi
#             bs = 1
#             feat_dim = data.mesh_feat.shape[-1]
#             # mesh_feat [coord_x, coord_y, u, hessian_norm]
#             node_num = data.mesh_feat.view(bs, -1, feat_dim).shape[1]

#             out_area = get_face_area(output_coord, data.face)
#             in_area = get_face_area(data.x[:, :2], data.face)
#             # restore the sign of the area
#             out_area = torch.sign(in_area) * out_area
#             # mask for negative area
#             neg_mask = out_area < 0
#             neg_area = out_area[neg_mask]
#             # calculate the loss, we want it normalized by the batch size
#             # and loss should be positive, so we are using -1 here.
#             num_tangle += len(neg_area)
#         return num_tangle / len(dataset)

#     # deprecated, do not use this option unless you know what you are doing
#     elif (method == "msg"):
#         for i in range(len(dataset)):
#             data = dataset[i].to(device)
#             with torch.no_grad():
#                 output_data = model(data)
#                 input_edge = data.edge_index
#                 mesh = data.x[:, :2]
#                 mesh_new = output_data
#                 Counter = TangleCounter()
#                 num_tangle += Counter(mesh, mesh_new, input_edge).item()
#         num_tangle = num_tangle / len(dataset)
#         return num_tangle



def print_parameter_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                print(f'param name: {name}, grad: {torch.sum(param.grad)}')
            else:
                print(f'param name: {name}, grad is None!')



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
        

        print_parameter_grad(model)

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


def interpolate(u, ori_mesh_x, ori_mesh_y, moved_x, moved_y):
    """
    u: [bs, node_num, 1]
    ori_mesh_x: [bs, node_num, 1]
    ori_mesh_y: [bs, node_num, 1]
    moved_x: [bs, node_num, 1]
    moved_y: [bs, node_num, 1]

    Note: node_num equals to sample_num
    """
    batch_size = u.shape[0]
    sample_num = u.shape[1]
    # print(f"batch size: {batch_size}, sample num: {sample_num}")
    u_interpolateds = []
    for bs in range(batch_size):
        # For a sample point of interest, we need to do a weighted summation over all other sample points
        # To avoid using a loop, we expand an additonal dim of size sample_num
        original_mesh = torch.cat((ori_mesh_x[bs], ori_mesh_y[bs]), dim=-1)
        moved_mesh = torch.cat((moved_x[bs], moved_y[bs]), dim=-1).unsqueeze(-2).repeat(1, sample_num, 1)
        # print(f"new mesh shape {moved_mesh.shape}, original mesh shape {original_mesh.shape}")
        # print((moved_mesh - original_mesh),(moved_mesh - original_mesh).shape)
        # print("check dimension ", (moved_mesh - original_mesh)[:, 0])

        # The second dimension of distance is the different sample points
        distance = -torch.norm(moved_mesh - original_mesh, dim=-1) * np.sqrt(sample_num)
        # print('raw distance ', torch.norm(moved_mesh - original_mesh, dim=-1))
        # print('distance ', torch.norm(moved_mesh - original_mesh, dim=-1)* np.sqrt(sample_num))
        normalize = nn.Softmax(dim=-1)
        weight = normalize(distance)
        # print('weight shape ', weight.shape, u[bs].shape)
        # print('weight ', weight, u, u[bs].permute(1, 0) * weight)
        # print(u.shape, weight.shape)
        u_interpolateds.append(torch.sum(u[bs].permute(1, 0) * weight, dim=-1).unsqueeze(-1))
        # print(f"interpolated shape: {u_interpolateds[-1]}")
        # print('inte ', u_interpolated)
    return torch.stack(u_interpolateds, dim=0)


def _generate_samples(num_meshes, num_samples_per_mesh, coords, solution, monitor, redundant_sample_ratio=10, device='cuda'):
    meshes = torch.tensor(np.random.uniform(0, 1, (num_meshes, redundant_sample_ratio * num_samples_per_mesh, 2)), dtype=torch.float).to(device)
    solution_input = solution.repeat(num_meshes, 1, 1)
    monitor_input = monitor.repeat(num_meshes, 1, 1)
    coords_x = coords[: ,: ,0].unsqueeze(-1).repeat(num_meshes, 1, 1)
    coords_y = coords[: ,: ,1].unsqueeze(-1).repeat(num_meshes, 1, 1)
    new_meshes_x = meshes[:, :, 0].unsqueeze(-1)
    new_meshes_y = meshes[:, :, 1].unsqueeze(-1)

    solutions = interpolate(solution_input, coords_x, coords_y, new_meshes_x, new_meshes_y)
    monitors = interpolate(monitor_input, coords_x, coords_y, new_meshes_x, new_meshes_y)

    meshes_ = []
    soluitons_ = []
    monitors_ = []

    # resample according to the monitor values
    for bs in range(monitors.shape[0]):
        prob = monitors[bs, :, 0] / torch.sum(monitors[bs, :, 0])
        index = np.random.choice(a=meshes.shape[1], size=num_samples_per_mesh, replace=False, p=prob.cpu().numpy())
        # print(torch.max(prob), torch.min(prob), torch.max(monitors), torch.min(monitors))
        meshes_.append(meshes[bs, index, :])
        soluitons_.append(solutions[bs, index, :])
        monitors_.append(monitors[bs, index, :])
    return torch.stack(meshes_, dim=0), torch.stack(soluitons_, dim=0), torch.stack(monitors_, dim=0)


def generate_samples(bs, num_samples_per_mesh, data, num_meshes=5, device="cuda"):
    # num_meshes = 5
    # num_nodes = coord_ori.shape[-2] // bs
    # samples_q = data.mesh_feat[:, :4].view(bs, -1, 4)
    meshes_collection = []
    solutions_collection = []
    monitors_collection = []
    for b in range(bs):
        coords = data.mesh_feat[:, :2].view(bs, -1, 2)[b, :, :].view(1, -1, 2)
        solution = data.mesh_feat[:, 2].view(bs, -1, 1)[b, :, :].view(1, -1, 1)
        monitor = data.mesh_feat[:, 3].view(bs, -1, 1)[b, :, :].view(1, -1, 1)
        meshes, solutions, monitors = _generate_samples(num_meshes=num_meshes, num_samples_per_mesh=num_samples_per_mesh, coords=coords, solution=solution, monitor=monitor, device=device)
        # print(f"output meshes: {meshes.shape} solutions: {solutions.shape} monitor: {monitors.shape}")
        # merge the addtional sampled attributes (mesh, solution, monitor) to a large graph within one sample
        meshes_collection.append(torch.cat([coords.view(-1, 2), meshes.view(-1, 2)], dim=0)) 
        solutions_collection.append(torch.cat([solution.view(-1, 1), solutions.view(-1, 1)], dim=0))
        monitors_collection.append(torch.cat([monitor.view(-1, 1), monitors.view(-1, 1)], dim=0))
    
    # merge all enhanced sampled samples along the batch size dimension
    meshes_input = torch.stack(meshes_collection, dim=0)
    solutions_input = torch.stack(solutions_collection, dim=0)
    monitors_input = torch.stack(monitors_collection, dim=0)
    # merge all batched sampled attributes along feature dim for transformer key and value
    samples_kv = torch.cat([meshes_input, solutions_input, monitors_input], dim=-1)
    return samples_kv


def construct_graph(sampled_coords, num_neighbors=6):
    bs = sampled_coords.shape[0]
    num_per_mesh = sampled_coords.shape[1]
    batch = torch.tensor([x for x in range(bs)]).unsqueeze(-1).repeat(1, num_per_mesh).reshape(-1)
    edge_index = knn_graph(sampled_coords.view(-1, 2), k=num_neighbors, batch=batch, loop=False)
    return edge_index


def compute_phi_hessian(mesh_query_x, mesh_query_y, phix, phiy, out_monitor, bs, data, loss_func):
    feat_dim = data.mesh_feat.shape[-1]
    node_num = data.mesh_feat.view(bs, -1, feat_dim).shape[1]
    sampled_num = mesh_query_x.shape[0] // bs
    sampled_ratio = sampled_num // node_num

    # equation residual loss
    loss_eq_residual = torch.tensor(0.0)
    # Convex loss
    loss_convex = torch.tensor(0.0)
    if phix is not None and phiy is not None:
        # print(f"phix: {phix.shape}, phiy: {phiy.shape}")
        hessian_seed = torch.ones(phix.shape).to(device)
        phixx = torch.autograd.grad(phix, mesh_query_x, grad_outputs=hessian_seed, retain_graph=True, create_graph=True, allow_unused=True)[0]
        phixy = torch.autograd.grad(phix, mesh_query_y, grad_outputs=hessian_seed, retain_graph=True, create_graph=True, allow_unused=True)[0]
        phiyx = torch.autograd.grad(phiy, mesh_query_x, grad_outputs=hessian_seed, retain_graph=True, create_graph=True, allow_unused=True)[0]
        phiyy = torch.autograd.grad(phiy, mesh_query_y, grad_outputs=hessian_seed, retain_graph=True, create_graph=True, allow_unused=True)[0]

        # print(f"phix grad: {phix_grad.shape}, phiy grad: {phiy_grad.shape}")
        # phixx = phix_grad[:, 0]
        # phixy = phix_grad[:, 1]
        # phiyx = phiy_grad[:, 0]
        # phiyy = phiy_grad[:, 1]
        # print(f"phixx grad: {torch.sum(phixx)}, phixy grad: {torch.sum(phixy)}, phiyx grad: {torch.sum(phiyx)}, phiyy grad: {torch.sum(phiyy)}")
        det_hessian = (phixx + 1) * (phiyy + 1) - phixy * phiyx
        det_hessian = det_hessian.view(bs, sampled_num, 1)

        # jacobian_x = data.mesh_feat[:, 4].view(bs, node_num, 1)
        # jacobian_y = data.mesh_feat[:, 5].view(bs, node_num, 1)

        hessian_norm = data.mesh_feat[:, 3].view(bs, node_num, 1).repeat(1, sampled_ratio, 1)
        # solution = data.mesh_feat[:, 1].resahpe(bs, node_num, 1)
        # original_mesh_x = data.mesh_feat[:, 0].view(bs, node_num, 1)
        # original_mesh_y = data.mesh_feat[:, 1].view(bs, node_num, 1)

        moved_x = phix.view(bs, sampled_num, 1) + mesh_query_x.view(bs, sampled_num, 1)
        moved_y = phiy.view(bs, sampled_num, 1) + mesh_query_y.view(bs, sampled_num, 1)

        # print(f"diff x:{torch.abs(original_mesh_x - moved_x).mean()}, diff y:{torch.abs(original_mesh_y - moved_y).mean()}")
        # Interpolate on new moved mesh

        hessian_norm_ = interpolate(hessian_norm, mesh_query_x.view(bs, sampled_num, 1), mesh_query_y.view(bs, sampled_num, 1), moved_x, moved_y)
        enhanced_hessian_norm = hessian_norm_ #+ out_monitor.view(bs, node_num, 1)

        # =========================== jacobian related attempts ==================
        # jac_x = interpolate(jacobian_x, original_mesh_x, original_mesh_y, moved_x, moved_y)
        # jac_y = interpolate(jacobian_y, original_mesh_x, original_mesh_y, moved_x, moved_y)
        # # alpha = torch.sum(torch.sqrt(torch.abs(jac_x)**2 + torch.abs(jac_y)**2), dim=(-1, -2)) / node_num**2
        # alpha = 1.0

        # phixx = phixx.reshape(bs, node_num, 1)
        # phiyx = phiyx.reshape(bs, node_num, 1)
        # phiyy = phiyy.reshape(bs, node_num, 1)
        # phixy = phixy.reshape(bs, node_num, 1)

        # jac_xi_1 = jac_x * (1 + phixx) + jac_y * phiyx
        # jac_xi_2 = jac_x * phixy + jac_y * (1 + phiyy)
        # print(torch.sum(jac_xi_1), torch.sum(jac_xi_2))
        # monitor = monitor_grad(alpha, jac_xi_1, jac_xi_2) /1000
        # =========================== 

        lhs = enhanced_hessian_norm * det_hessian
        rhs = torch.sum(hessian_norm, dim=(1, 2)) / sampled_num
        rhs = rhs.unsqueeze(-1).repeat(1, sampled_num).unsqueeze(-1)
        loss_eq_residual = 1000 * loss_func(lhs, rhs)
        # print(torch.sum(hessian_norm_ - hessian_norm), hessian_norm_.shape, det_hessian.shape, lhs.shape, rhs.shape)
        # print(f"diff between interpolation jac x {torch.sum(jacobian_x - jac_x)} alpha: {alpha} monitor: {torch.sum(monitor)} det_hessian {torch.sum(det_hessian)} lhs {torch.sum(lhs)} rhs {torch.sum(rhs)}")

        # Convex loss
        # if use_convex_loss:
        loss_convex = torch.mean(torch.min(torch.tensor(0).type_as(phixx).to(device), 1 + phixx)**2 + torch.min(torch.tensor(0).type_as(phiyy).to(device), 1 + phiyy)**2)
        return loss_eq_residual, loss_convex, 


def model_forward(bs, data, model):
    # Create mesh query for deformer, seperate from the original mesh as feature for encoder 
    mesh_query_x = data.mesh_feat[:, 0].view(-1, 1).detach().clone()
    mesh_query_y = data.mesh_feat[:, 1].view(-1, 1).detach().clone()
    mesh_query_x.requires_grad = True
    mesh_query_y.requires_grad = True
    mesh_query = torch.cat([mesh_query_x, mesh_query_y], dim=-1)

    num_nodes = mesh_query.shape[-2] // bs
    # Generate random mesh queries for unsupervised learning
    sampled_queries = generate_samples(bs=bs, num_samples_per_mesh=num_nodes, num_meshes=5, data=data, device=device)
    sampled_queries_edge_index = construct_graph(sampled_queries[:, :, :2], num_neighbors=6)

    mesh_sampled_queries_x = sampled_queries[:, :, 0].view(-1, 1).detach()
    mesh_sampled_queries_y = sampled_queries[:, :, 1].view(-1, 1).detach()
    mesh_sampled_queries_x.requires_grad = True
    mesh_sampled_queries_y.requires_grad = True
    mesh_sampled_queries = torch.cat([mesh_sampled_queries_x, mesh_sampled_queries_y], dim=-1).view(-1, 2)

    coord_ori_x = data.mesh_feat[:, 0].view(-1, 1)
    coord_ori_y = data.mesh_feat[:, 1].view(-1, 1)
    coord_ori_x.requires_grad = True
    coord_ori_y.requires_grad = True
    coord_ori = torch.cat([coord_ori_x, coord_ori_y], dim=-1)

    num_nodes = coord_ori.shape[-2] // bs
    input_q = data.mesh_feat[:, :4]
    input_kv = generate_samples(bs=bs, num_samples_per_mesh=num_nodes, data=data, device=device)
    # print(f"batch size: {bs}, num_nodes: {num_nodes}, input q", input_q.shape, "input_kv ", input_kv.shape)

    (output_coord_all, output, out_monitor), (phix, phiy) = model(data, input_q, input_q, mesh_query, mesh_sampled_queries, sampled_queries_edge_index)
    # (output_coord_all, output, out_monitor), (phix, phiy) = model(data, input_q, input_kv, mesh_query, sampled_queries, sampled_queries_edge_index)
    output_coord = output_coord_all[:num_nodes*bs]
    # print(output_coord_all.shape, output_coord.shape)

    # mesh_query_x_all = torch.cat([mesh_query_x, mesh_sampled_queries[:, :, 0].view(-1, 1)], dim=0)
    # mesh_query_y_all = torch.cat([mesh_query_y, mesh_sampled_queries[:, :, 1].view(-1, 1)], dim=0)
    mesh_query_x_all = mesh_sampled_queries_x
    mesh_query_y_all = mesh_sampled_queries_y
    return output_coord, output, out_monitor, phix, phiy, mesh_query_x_all, mesh_query_y_all


def train_unsupervised(
        loader, model, optimizer, device, loss_func,
        use_jacob=False,
        use_inversion_loss=False,
        use_inversion_diff_loss=False,
        use_area_loss=False,
        use_convex_loss=False,
        weight_area_loss=1,
        weight_deform_loss=1,
        weight_eq_residual_loss=1,
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
    total_eq_residual_loss = 0
    total_convex_loss = 0
    total_deform_loss = 0
    total_inversion_loss = 0
    total_inversion_diff_loss = 0
    total_area_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        data = batch.to(device)

        output_coord, output, out_monitor, phix, phiy, mesh_query_x_all, mesh_query_y_all = model_forward(bs, data, model)
        loss_eq_residual, loss_convex = compute_phi_hessian(mesh_query_x_all, mesh_query_y_all, phix, phiy, out_monitor, bs, data, loss_func=loss_func)

        # loss_eq_residual, loss_convex = torch.tensor(0.0), torch.tensor(0.0)

        if not use_convex_loss:
            loss_convex = torch.tensor(0.0)

        loss = 0
        inversion_loss = 0
        deform_loss = torch.tensor(0.0)
        inversion_diff_loss = 0
        area_loss = 0
        # deformation loss
        deform_loss = 1000 * (
            loss_func(output_coord, data.y) if not use_jacob else
            jacobLoss(model, output_coord, data, loss_func)
        )
        # Inversion loss
        if use_inversion_loss:
            inversion_loss = get_inversion_loss(
                output_coord, data.y, data.face,
                batch_size=bs, scaler=scaler)
        # if use_area_loss:
        area_loss = get_area_loss(
            output_coord, data.y, data.face, bs, scaler)
        

        loss = (
            weight_deform_loss * deform_loss +
            inversion_loss +
            inversion_diff_loss +
            weight_area_loss * area_loss  + 
            weight_eq_residual_loss * loss_eq_residual +
            loss_convex
        )

        # Jacobian loss
        if use_jacob:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        
        # print_parameter_grad(model)

        optimizer.step()
        total_loss += loss.item()
        total_eq_residual_loss += loss_eq_residual.item()
        total_convex_loss += loss_convex.item() if use_convex_loss else 0
        total_deform_loss += deform_loss.item()
        total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0 # noqa
        total_inversion_diff_loss += inversion_diff_loss.item() if use_inversion_diff_loss else 0 # noqa
        total_area_loss += area_loss.item()

    res = {
        "total_loss": total_loss / len(loader),
        "deform_loss": total_deform_loss / len(loader),
        "equation_residual": total_eq_residual_loss / len(loader)
    }
    if (use_convex_loss):
        res["convex_loss"] = total_convex_loss / len(loader)
    if (use_inversion_loss):
        res["inversion_loss"] = total_inversion_loss / len(loader)
    if (use_inversion_diff_loss):
        res["inversion_diff_loss"] = total_inversion_diff_loss / len(loader)
    if (use_area_loss):
        res["area_loss"] = total_area_loss / len(loader)

    return res


def evaluate_unsupervised(
        loader, model, device, loss_func,
        use_jacob=False,
        use_inversion_loss=False,
        use_inversion_diff_loss=False,
        use_area_loss=False,
        use_convex_loss=False,
        weight_area_loss=1,
        weight_deform_loss=1,
        weight_eq_residual_loss=1,
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
    total_eq_residual_loss = 0
    total_convex_loss = 0
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

        # with torch.no_grad():

        output_coord, output, out_monitor, phix, phiy, mesh_query_x_all, mesh_query_y_all = model_forward(bs, data, model)
        loss_eq_residual, loss_convex = compute_phi_hessian(mesh_query_x_all, mesh_query_y_all, phix, phiy, out_monitor, bs, data, loss_func=loss_func)
        # loss_eq_residual, loss_convex = torch.tensor(0.0), torch.tensor(0.0)
        
        if not use_convex_loss:
            loss_convex = torch.tensor(0.0)

        deform_loss =  1000 * (
            loss_func(output_coord, data.y) if not use_jacob else
            jacobLoss(model, output_coord, data, loss_func)
        )
        inversion_loss = 0
        if use_inversion_loss:
            inversion_loss = get_inversion_loss(
                output_coord, data.y, data.face,
                batch_size=bs, scaler=scaler)
        # if use_area_loss:
        area_loss = get_area_loss(
                output_coord, data.y, data.face, bs, scaler)

        loss = (
            weight_deform_loss * deform_loss +
            inversion_loss +
            inversion_diff_loss +
            weight_area_loss * area_loss +
            weight_eq_residual_loss * loss_eq_residual +
            loss_convex
        )

        total_loss += loss.item()
        total_eq_residual_loss += loss_eq_residual.item()
        total_convex_loss += loss_convex.item() if use_convex_loss else 0
        total_deform_loss += deform_loss.item()
        total_inversion_diff_loss += inversion_diff_loss.item() if use_inversion_diff_loss else 0 # noqa
        total_inversion_loss += inversion_loss.item() if use_inversion_loss else 0  # noqa
        total_area_loss += area_loss.item()
    res = {
        "total_loss": total_loss / len(loader),
        "deform_loss": total_deform_loss / len(loader),
        "equation_residual": total_eq_residual_loss / len(loader)
    }
    if (use_convex_loss):
        res["convex_loss"] = total_convex_loss / len(loader)
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
        bs = loader.batch_size
        for data in loader:
            with torch.no_grad():
                data = data.to(device)
                # # Create mesh query for deformer, seperate from the original mesh as feature for encoder 
                # mesh_query_x = data.mesh_feat[:, 0].view(-1, 1).detach().clone()
                # mesh_query_y = data.mesh_feat[:, 1].view(-1, 1).detach().clone()
                # mesh_query_x.requires_grad = True
                # mesh_query_y.requires_grad = True
                # mesh_query = torch.cat([mesh_query_x, mesh_query_y], dim=-1)

                # num_nodes = mesh_query.shape[-2] // bs
                # # Generate random mesh queries for unsupervised learning
                # sampled_queries = generate_samples(bs=bs, num_samples_per_mesh=num_nodes, data=data, num_meshes=1, device=device)
                # sampled_queries_edge_index = construct_graph(sampled_queries[:, :, :2])

                # mesh_sampled_queries_x = sampled_queries[:, :, 0].view(-1, 1).detach()
                # mesh_sampled_queries_y = sampled_queries[:, :, 1].view(-1, 1).detach()
                # mesh_sampled_queries_x.requires_grad = True
                # mesh_sampled_queries_y.requires_grad = True
                # mesh_sampled_queries = torch.cat([mesh_sampled_queries_x, mesh_sampled_queries_y], dim=-1).view(-1, 2)


                # coord_ori_x = data.mesh_feat[:, 0].view(-1, 1)
                # coord_ori_y = data.mesh_feat[:, 1].view(-1, 1)
                # coord_ori_x.requires_grad = True
                # coord_ori_y.requires_grad = True
                # coord_ori = torch.cat([coord_ori_x, coord_ori_y], dim=-1)

                # num_nodes = coord_ori.shape[-2] // bs
                # input_q = torch.cat([mesh_query, data.mesh_feat[:, 2:4]], dim=-1)
                # input_kv = generate_samples(bs=bs, num_samples_per_mesh=num_nodes, data=data, device=device)
                # # print(f"batch size: {bs}, num_nodes: {num_nodes}, input q", input_q.shape, "input_kv ", input_kv.shape)

                # (output_coord_all, output, out_monitor), (phix, phiy) = model(data.to(device), input_q.to(device), input_q.to(device), mesh_query.to(device), mesh_sampled_queries.to(device), sampled_queries_edge_index)
                # # (output_coord_all, output, out_monitor), (phix, phiy) = model(data, input_q, input_kv, mesh_query, sampled_queries, sampled_queries_edge_index)
                # output_data = output_coord_all[:num_nodes*bs]
                output_coord, output, out_monitor, phix, phiy, mesh_query_x_all, mesh_query_y_all = model_forward(bs, data, model)

                out_area = get_face_area(output_coord, data.face)
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


def load_model(model, weight_path, strict=False):
    """
    Loads pre-trained weights into a PyTorch model from a given file path.

    Args:
        model (torch.nn.Module): The PyTorch model.
        weight_path (str): File path to the pre-trained model weights.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    try:
        model.load_state_dict(torch.load(weight_path), strict=strict)
    except RuntimeError:
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=strict)
    return model
