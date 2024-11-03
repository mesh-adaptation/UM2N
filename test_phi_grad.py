import warnings
from types import SimpleNamespace

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import SequentialSampler
from torch_geometric.data import DataLoader

import UM2N
from UM2N.model.train_util import (
    generate_samples_structured_grid,
    model_forward,
)

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# run_id = 'welbby7t'
# run_id = 'vwopbol5'
# run_id = 'ixtqouzl'
# run_id = '0iwpdpnr' # MRN
run_id = "mfn1hnrg"  # MRT
run_id = "uu515eu1"  # MRN-LTE
run_id = "ywtfui2q"  # MRN-GTE
run_id = "gboubixk"  # M2T
run_id = "xqa8fnoj"  # M2N
# run_id = 'l9cfh1wj' # MRT
# run_id = 'j9rjsxl1' # MRT + sampling
# run_id = 'hegubzg0' # MRN + sampling
run_id = "2b0ouh5p"  # MRT + no up bottom left right

run_id = "kg4y9zak"  # MRT + mask 0.95
run_id = "lmcata0v"  # MRT + mask 0.75
run_id = "8becmygf"  # MRT + mask 0.50
run_id = "n2qcocej"  # MRT + mask 0.25

run_id = "esn5ynfq"  # MRT + 1 layer recurrent attention mask 0.50
run_id = "7yaerq40"  # MRT + 1 layer recurrent attention mask 0.50~0.90

run_id = "lvcal7vq"  # MRT + 1 layer recurrent + mask 0.5
run_id = "t233y3ik"  # MRT + 1 layer recurrent + mask 0.75
run_id = "yl8fmiip"  # MRT + 1 layer recurrent + mask 0.95
run_id = "nevv2a0d"  # MRT + 1 layer recurrent + mask 0.5 ~ 0.9

run_id = "kx25grpm"  # MRT + 3 layers recurrent
run_id = "790xybc1"  # MRT + 2 layers recurrent
run_id = "zdj9ocmw"  # MRT + 1 layer recurrent

run_id = "2wta7yed"  # MRT-1R with sampling
run_id = "0bsy6m45"  # MRT-1R no hessian


run_id = "fzgaycnv"  # MRT-1R output phi

run_id = "mug27xhl"  # MRT-1R output phi convex loss


run_id = "9ygg08yg"  # MRT-1R output phi, constrain bd

run_id = "u14bt77h"  # output phi grad

run_id = "f4q1v2pd"  # output coord

run_id = "kst5ig88"  # output phi grad large eq residual


run_id = "c2kyy4vl"  # purely unsupervised
run_id = "a2af7x3j"  # weight_d = 0.01 weight_u


run_id = "bzlj9vcl"  # unsupervised 1 1 1

run_id = "8ndi2teh"  # unsupervised 1 1 1, small unsupervised


run_id = "2f4florr"  # unsupervised 1 01 01,  large unsupervised

run_id = "fhmqq8eg"  # supervised phi grad test

run_id = "s6qrcn54"  # unsupervised 1 1 1, large unsupervised


run_id = "kuz2edst"  # refactor the gradient

run_id = "yn3aaiwi"  # mesh query semi 111
run_id = "pk66tmjj"  # mesh query semi 111 50 smaples

run_id = "0l8ujpdr"  # mesh query semi 111, old dataset


run_id = "zae8jkpm"  # mesh query semi 111


run_id_collections = {
    "MRT": ["mfn1hnrg"],
    "MRT-no-udlr": ["2b0ouh5p"],
    "MRT-1R-phi": ["fzgaycnv"],
    "MRT-1R-phi-convex": ["mug27xhl"],
    "MRT-1R-phi-bd": ["9ygg08yg"],
    "MRT-1R-phi-grad": ["u14bt77h"],
    "MRT-1R-phi-grad-eq": ["kst5ig88"],
    "MRT-1R-coord": ["f4q1v2pd"],
    "MRT-1R-phi-grad-un": ["c2kyy4vl"],
    "MRT-1R-phi-grad-quasi-un": ["a2af7x3j"],
    "MRT-1R-phi-grad-un-111": ["bzlj9vcl"],
    "MRT-1R-phi-grad-un-111-small": ["8ndi2teh"],
    "MRT-1R-phi-grad-un-111-large": ["2f4florr"],
    "MRT-1R-phi-grad-test": ["fhmqq8eg"],
    "MRT-1R-phi-grad-un-grad-test": ["s6qrcn54"],
    "MRT-1R-phi-grad-un-grad-test-new": ["kuz2edst"],
    # "MRT-1R-phi-grad-un-grad-test-query": ['zae8jkpm'],
    # "MRT-1R-phi-grad-un-grad-test-query": ['0l8ujpdr'],
    "MRT-1R-phi-grad-un-grad-test-query": ["8ndi2teh"],
    "MRT-1R": ["zdj9ocmw"],
    "MRT-2R": ["790xybc1"],
    "MRT-3R": ["kx25grpm"],
    "MRT-1R-sampling": ["2wta7yed"],
    "MRT-1R-no-hessian": ["0bsy6m45"],
    "MRT-1R-atten-mask0.50": ["esn5ynfq"],
    "MRT-1R-atten-mask0.50~0.90": ["7yaerq40"],
    "MRT-1R-mask0.95": ["yl8fmiip"],
    "MRT-1R-mask0.75": ["t233y3ik"],
    "MRT-1R-mask0.50": ["lvcal7vq"],
    "MRT-1R-mask0.50~0.9": ["nevv2a0d"],
    "MRT-mask0.95": ["kg4y9zak"],
    "MRT-mask0.75": ["lmcata0v"],
    "MRT-mask0.50": ["8becmygf"],
    "MRT-mask0.25": ["n2qcocej"],
    "MRT-Sampling": ["j9rjsxl1"],
    "MRN-Sampling": ["hegubzg0"],
    "MRN-GTE": ["ywtfui2q"],
    "MRN-LTE": ["uu515eu1"],
    "MRN": ["0iwpdpnr"],
    "M2T": ["gboubixk"],
    "M2N": ["xqa8fnoj"],
}

dataset_name = "helmholtz"
# dataset_name = 'swirl'

# test_ms = 'poly'
test_ms = 20
num_sample_vis = 5
# models_to_compare = ["MRT", "MRN-LTE", "MRT-Sampling", "MRN-Sampling", "MRN", "M2T", "M2N"]
# models_to_compare = ["MRT", "MRT-mask0.75", "MRT-mask0.50", "MRT-mask0.25", "MRN-LTE", "MRN", "M2T", "M2N"]
# models_to_compare = ["MRT", "MRT-mask0.75", "MRT-mask0.50", "MRT-mask0.25"]
# models_to_compare = ["MRT", "MRT-1R", "MRT-1R-atten-mask0.50", "MRT-1R-atten-mask0.50~0.90", "MRT-1R-mask0.50", "MRT-1R-mask0.50~0.9"]

# models_to_compare = ["MRT-no-udlr", "MRT-no-udlr"]
# models_to_compare = ["MRT-1R-phi", "MRT-1R-phi-bd"]
# models_to_compare = ["MRT-1R-phi-grad-un-111", "MRT-1R-phi-bd", "MRT-1R-coord"]

# models_to_compare = ["MRT-1R", "MRT-1R-no-hessian"]
# test dataset, for benchmarking loss effects on model performance


models_to_compare = ["MRT-1R-phi-grad-un-111-small"]
models_to_compare = ["MRT-1R-phi-grad-un-grad-test-new"]

models_to_compare = ["MRT-1R-phi-grad-un-grad-test-query"]
# models_to_compare = ["MRT-1R-phi-grad-un-grad-test"]
# models_to_compare = ["MRT-1R-phi-grad"]

if dataset_name == "helmholtz":
    # test_dir = f"./data/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"

    # dataset_dir = f"./data/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=10_aniso_full"
    # dataset_dir = f"./data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full"
    dataset_dir = "./data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=50_aniso_full_algo_6"
    # dataset_dir = f"./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6"
    # test_dir = f"./data/with_sampling/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"
    # test_dir = f"./data/large_scale_test/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"
    # test_dir = f"./data/helmholtz_poly/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.06_n=400_aniso_full/data"
elif dataset_name == "swirl":
    # dataset_dir = f"./data/dataset/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5"
    # Swirl
    dataset_dir = "./data/swirl/z=<0,1>_ndist=None_max_dist=6_<30x30>_n=iso_pad"
test_dir = f"{dataset_dir}/data"

random_seed = 1


# Collect from dataset
phi_MA_collections = {}
phix_MA_collections = {}
phiy_MA_collections = {}
phixx_MA_collections = {}
phixy_MA_collections = {}
phiyx_MA_collections = {}
phiyy_MA_collections = {}

# Collect from model
out_mesh_collections = {}
out_phix_collections = {}
out_phiy_collections = {}

# Finite difference grad
out_phixx_collections = {}
out_phixy_collections = {}
out_phiyy_collections = {}
out_phiyx_collections = {}

# Ad grad
out_ad_phixx_collections = {}
out_ad_phixy_collections = {}
out_ad_phiyy_collections = {}
out_ad_phiyx_collections = {}

out_loss_collections = {}
out_atten_collections = {}
for model_name in models_to_compare:
    run_id = run_id_collections[model_name][0]
    entity = "mz-team"
    project_name = "warpmesh"

    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    print(config)
    config.num_transformer_in = 4
    if "no-hessian" in model_name:
        config.num_transformer_in = 3
    config.num_transformer_out = 16
    config.num_transformer_embed_dim = 64
    config.num_transformer_heads = 4
    config.num_transformer_layers = 1
    if run_id == "9ygg08yg":
        config.deform_out_type = "phi"

    model = None
    if config.model_used == "MRTransformer":
        model = UM2N.MRTransformer(
            num_transformer_in=config.num_transformer_in,
            num_transformer_out=config.num_transformer_out,
            num_transformer_embed_dim=config.num_transformer_embed_dim,
            num_transformer_heads=config.num_transformer_heads,
            num_transformer_layers=config.num_transformer_layers,
            deform_in_c=config.num_deform_in,
            deform_out_type=config.deform_out_type,
            num_loop=config.num_deformer_loop,
            device=device,
        )
    else:
        raise Exception(f"Model {config.model_used} not implemented.")
    # config.mesh_feat.extend(['grad_u', 'phi', 'grad_phi', 'jacobian'])
    if "grad_u" not in config.mesh_feat:
        config.mesh_feat.extend(["grad_u", "phi", "grad_phi", "jacobian"])
    else:
        config.mesh_feat.extend(["phi", "grad_phi", "jacobian"])
    print("mesh feat type ", config.mesh_feat)
    test_set = UM2N.MeshDataset(
        test_dir,
        # transform=UM2N.normalise if UM2N.normalise else None,
        transform=None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        use_cluster=config.use_cluster,
        load_jacobian=True,
    )
    print("mesh feat mesh feat ", test_set[0].mesh_feat.shape)
    loader = DataLoader(
        test_set,
        # batch_size=1,
        sampler=SequentialSampler(test_set),
    )
    # shuffle=False)

    epoch = 999
    # TODO: the MRN-Sampling ('hegubzg0') only trained 800 epochs
    if run_id == "hegubzg0":
        epoch = 799

    target_file_name = "model_{}.pth".format(epoch)

    model_file = None
    for file in run.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(root=".", replace=True)
            print("download file ", model_file)

    model_file = None
    if model_file is None:
        print("No model file found on wandb! Load the local backup.")
        model_file = f"./out/{config.experiment_name}/{target_file_name}"
        target_file_name = model_file
    assert model_file is not None, "Model file not found either on wandb or local."
    print(target_file_name)
    model = UM2N.load_model(model, model_file)
    print(model)

    loss_func = torch.nn.L1Loss()
    model.to(device)
    model.eval()

    out_mesh_collections[model_name] = []
    out_loss_collections[model_name] = []
    out_atten_collections[model_name] = []
    out_phix_collections[model_name] = []
    out_phiy_collections[model_name] = []
    out_phixx_collections[model_name] = []
    out_phixy_collections[model_name] = []
    out_phiyy_collections[model_name] = []
    out_phiyx_collections[model_name] = []
    out_ad_phixx_collections[model_name] = []
    out_ad_phixy_collections[model_name] = []
    out_ad_phiyy_collections[model_name] = []
    out_ad_phiyx_collections[model_name] = []

    phi_MA_collections[model_name] = []
    phix_MA_collections[model_name] = []
    phiy_MA_collections[model_name] = []
    phixx_MA_collections[model_name] = []
    phixy_MA_collections[model_name] = []
    phiyx_MA_collections[model_name] = []
    phiyy_MA_collections[model_name] = []

    target_mesh = []
    target_face = []
    target_hessian_norm = []
    num_step_recurrent = 5
    # with torch.no_grad():
    cnt = 0
    torch.manual_seed(random_seed)
    start_num = 0
    # for batch in loader:
    for i in range(start_num, start_num + num_sample_vis):
        batch = test_set[i]
        batch.conv_feat = batch.conv_feat.unsqueeze(0)
        batch.conv_feat_fix = batch.conv_feat_fix.unsqueeze(0)
        # print(batch)
        sample = batch.to(device)
        print("mesh feat shape ", sample.mesh_feat.shape)
        if model_name == "MRT":
            out = model.move(sample, num_step=5)
        else:
            if "MRT-1R" in model_name:
                if "phi" in model_name:
                    # mesh feat: [coord_x, coord_y, u, hessian_norm, grad_u_x, grad_u_y, phi, grad_phi_x, grad_phi_y, hessian_phi(4 elements)]
                    phi_MA_collections[model_name].append(
                        sample.mesh_feat[:, 6].detach().cpu().numpy()
                    )
                    phix_MA_collections[model_name].append(
                        sample.mesh_feat[:, 7].detach().cpu().numpy()
                    )
                    phiy_MA_collections[model_name].append(
                        sample.mesh_feat[:, 8].detach().cpu().numpy()
                    )

                    phixx_MA_collections[model_name].append(
                        sample.mesh_feat[:, 9].detach().cpu().numpy()
                    )
                    phixy_MA_collections[model_name].append(
                        sample.mesh_feat[:, 10].detach().cpu().numpy()
                    )
                    phiyx_MA_collections[model_name].append(
                        sample.mesh_feat[:, 11].detach().cpu().numpy()
                    )
                    phiyy_MA_collections[model_name].append(
                        sample.mesh_feat[:, 12].detach().cpu().numpy()
                    )

                    # Mannually normlize for model input
                    mesh_val_feat = sample.mesh_feat[:, 2:]  # value feature (no coord)
                    min_val = torch.min(mesh_val_feat, dim=0).values
                    max_val = torch.max(mesh_val_feat, dim=0).values
                    max_abs_val = torch.max(torch.abs(min_val), torch.abs(max_val))
                    sample.mesh_feat[:, 2:] = sample.mesh_feat[:, 2:] / max_abs_val

                    # Create mesh query for deformer, seperate from the original mesh as feature for encoder
                    # mesh_query_x = sample.mesh_feat[:, 0].view(-1, 1).detach().clone()
                    # mesh_query_y = sample.mesh_feat[:, 1].view(-1, 1).detach().clone()
                    # mesh_query_x.requires_grad = True
                    # mesh_query_y.requires_grad = True
                    # mesh_query = torch.cat([mesh_query_x, mesh_query_y], dim=-1)

                    # coord_ori_x = sample.mesh_feat[:, 0].view(-1, 1)
                    # coord_ori_y = sample.mesh_feat[:, 1].view(-1, 1)

                    # coord_ori_x.requires_grad = True
                    # coord_ori_y.requires_grad = True

                    # coord_ori = torch.cat([coord_ori_x, coord_ori_y], dim=-1)

                    # (out, phi, out_monitor), (phix, phiy) = model.move(sample, coord_ori, mesh_query, num_step=1)
                    # (out, phi, out_monitor), (phix, phiy) = model(sample, coord_ori, mesh_query)
                    bs = 1
                    (
                        output_coord,
                        output,
                        out_monitor,
                        phix,
                        phiy,
                        mesh_query_x_all,
                        mesh_query_y_all,
                    ) = model_forward(bs, sample, model, use_add_random_query=False)

                    feat_dim = sample.mesh_feat.shape[-1]
                    node_num = sample.mesh_feat.reshape(1, -1, feat_dim).shape[1]

                    out_phix_collections[model_name].append(phix.detach().cpu().numpy())
                    out_phiy_collections[model_name].append(phiy.detach().cpu().numpy())

                    hessian_seed = torch.ones(phix.shape).to(device)
                    phixx_ad = torch.autograd.grad(
                        phix,
                        mesh_query_x_all,
                        grad_outputs=hessian_seed,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    phixy_ad = torch.autograd.grad(
                        phix,
                        mesh_query_y_all,
                        grad_outputs=hessian_seed,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    phiyx_ad = torch.autograd.grad(
                        phiy,
                        mesh_query_x_all,
                        grad_outputs=hessian_seed,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    phiyy_ad = torch.autograd.grad(
                        phiy,
                        mesh_query_y_all,
                        grad_outputs=hessian_seed,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]

                    out_ad_phixx_collections[model_name].append(
                        phixx_ad.detach().cpu().numpy()
                    )
                    out_ad_phiyy_collections[model_name].append(
                        phiyy_ad.detach().cpu().numpy()
                    )
                    out_ad_phixy_collections[model_name].append(
                        phixy_ad.detach().cpu().numpy()
                    )
                    out_ad_phiyx_collections[model_name].append(
                        phiyx_ad.detach().cpu().numpy()
                    )

                    mesh_query_x_all = mesh_query_x_all.view(bs, -1, 1)
                    mesh_query_y_all = mesh_query_y_all.view(bs, -1, 1)
                    _, _, _, _, phixy, phixx = generate_samples_structured_grid(
                        torch.cat([mesh_query_x_all, mesh_query_y_all], dim=-1), phix
                    )
                    _, _, _, _, phiyy, phiyx = generate_samples_structured_grid(
                        torch.cat([mesh_query_x_all, mesh_query_y_all], dim=-1), phiy
                    )

                    out_phixx_collections[model_name].append(
                        phixx.detach().cpu().numpy()
                    )
                    out_phiyy_collections[model_name].append(
                        phiyy.detach().cpu().numpy()
                    )
                    out_phixy_collections[model_name].append(
                        phixy.detach().cpu().numpy()
                    )
                    out_phiyx_collections[model_name].append(
                        phiyx.detach().cpu().numpy()
                    )

                    out = output_coord[: node_num * bs]

                elif "MRT-1R-coord" in model_name:
                    out, (phix, phiy) = model.move(sample, num_step=1)
                else:
                    out = model.move(sample, num_step=1)
            elif "MRT-2R" in model_name:
                out = model.move(sample, num_step=2)
            elif "MRT-3R" in model_name:
                out = model.move(sample, num_step=3)
            else:
                out = model.move(sample, num_step=5)
        print(out.shape)
        # if 'MRT' in model_name:
        #     attentions = model.get_attention_scores(sample)
        deform_loss = loss_func(out, sample.y) * 1000
        print(
            f"{model_name} {cnt} deform loss: {deform_loss}, mesh vertices: {out.shape}"
        )
        out_mesh_collections[model_name].append(out.detach().cpu().numpy())
        out_loss_collections[model_name].append(deform_loss)
        # out_atten_collections[model_name].append(attentions)
        target_mesh.append(sample.y.detach().cpu().numpy())
        target_face.append(sample.face.detach().cpu().numpy())
        target_hessian_norm.append(sample.mesh_feat[:, -1].detach().cpu().numpy())
        # compare_fig = UM2N.plot_mesh_compare(
        #     out.detach().cpu().numpy(), sample.y,
        #     sample.face
        # )
        # compare_fig.savefig(f"./out_images/img_method_{config.model_used}_reso_{test_ms}_{cnt}.png")
        cnt += 1
        if cnt == num_sample_vis:
            break


# compare_fig = UM2N.plot_multiple_mesh_compare(out_mesh_collections, out_loss_collections, target_mesh, target_face)
# compare_fig.tight_layout()
# compare_fig.subplots_adjust(top=0.95)
# compare_fig.suptitle(f"{dataset_name}: Output Mesh Comparsion (mesh resolution {test_ms}, dataloder seed: {random_seed})", fontsize=24)
# compare_fig.savefig(f"./out_images/{dataset_name}_comparison_reso_{test_ms}_seed_{random_seed}_recurrent_{num_step_recurrent}.png")

# out_tri = model(in_data.to(device))

# phix_grad = out_phix_collections["MRT-1R-phi-grad-un-111"][0]
# phiy_grad = out_phiy_collections["MRT-1R-phi-grad-un-111"][0]

# phixx = out_phixx_collections["MRT-1R-phi-grad-un-111"][0]
# phiyy = out_phiyy_collections["MRT-1R-phi-grad-un-111"][0]

# coord = out_mesh_collections["MRT-1R-phi-grad-un-111"][0]

##################################################################

# model_name = "MRT-1R-phi-grad-un-111-small"
# model_name = "MRT-1R-phi-grad-un-grad-test"
# model_name = "MRT-1R-phi-grad"
model_name = "MRT-1R-phi-grad-un-grad-test-new"
model_name = "MRT-1R-phi-grad-un-grad-test-query"
num_selected = 2

# Grad from finite difference
phix = out_phix_collections[model_name][num_selected]
phiy = out_phiy_collections[model_name][num_selected]
phixx = out_phixx_collections[model_name][num_selected]
phiyy = out_phiyy_collections[model_name][num_selected]
phixy = out_phixy_collections[model_name][num_selected]
phiyx = out_phiyx_collections[model_name][num_selected]

# Grad from ad
phixx_ad = out_ad_phixx_collections[model_name][num_selected]
phiyy_ad = out_ad_phiyy_collections[model_name][num_selected]
phixy_ad = out_ad_phixy_collections[model_name][num_selected]
phiyx_ad = out_ad_phiyx_collections[model_name][num_selected]

coord = out_mesh_collections[model_name][num_selected]
fd_coord = target_mesh[num_selected]

phi_sample = phi_MA_collections[model_name][num_selected]
phix_sample = phix_MA_collections[model_name][num_selected]
phiy_sample = phiy_MA_collections[model_name][num_selected]
phixx_sample = phixx_MA_collections[model_name][num_selected]
phixy_sample = phixy_MA_collections[model_name][num_selected]
phiyx_sample = phiyx_MA_collections[model_name][num_selected]
phiyy_sample = phiyy_MA_collections[model_name][num_selected]

import os

variables_collections = {
    r"$\nabla_x \phi$": (phix, phix, phix_sample),
    r"$\nabla_y \phi$": (phiy, phiy, phiy_sample),
    r"$\nabla_{xx} \phi$": (
        phixx,
        phixx_ad,
        phixx_sample - 1,
    ),  # The values recored from firedrake is actually I + H(\phi)
    r"$\nabla_{xy} \phi$": (phixy, phixy_ad, phixy_sample),
    r"$\nabla_{yx} \phi$": (phiyx, phiyx_ad, phiyx_sample),
    r"$\nabla_{yy} \phi$": (phiyy, phiyy_ad, phiyy_sample - 1),
}

num_variables = len(variables_collections.keys())

font_size = 24
mesh_gen = UM2N.UnstructuredSquareMesh()

if dataset_name == "helmholtz":
    model_mesh = mesh_gen.load_mesh(
        file_path=os.path.join(f"{dataset_dir}/mesh", f"mesh{num_selected}.msh")
    )
    fd_mesh = mesh_gen.load_mesh(
        file_path=os.path.join(f"{dataset_dir}/mesh", f"mesh{num_selected}.msh")
    )
elif dataset_name == "swirl":
    model_mesh = mesh_gen.load_mesh(
        file_path=os.path.join(f"{dataset_dir}/mesh", "mesh.msh")
    )
    fd_mesh = mesh_gen.load_mesh(
        file_path=os.path.join(f"{dataset_dir}/mesh", "mesh.msh")
    )

fig, axs = plt.subplots(num_variables, 6, figsize=(48, 8 * num_variables))

row = 0
for name, (finite_diff_val, model_val, fd_val) in variables_collections.items():
    # model values
    model_mesh.coordinates.dat.data[:] = coord[:]
    mesh_function_space = fd.Function(fd.FunctionSpace(model_mesh, "CG", 1))
    mesh_function_space.dat.data[:] = model_val.reshape(-1)[:]
    plt_obj3 = fd.tripcolor(mesh_function_space, axes=axs[row, 0])
    axs[row, 0].set_title(
        f"{dataset_name} - {name} (torch autograd)", fontsize=font_size
    )
    plt.colorbar(plt_obj3)

    # Finite difference grad
    mesh_function_space.dat.data[:] = finite_diff_val.reshape(-1)[:]
    plt_obj3 = fd.tripcolor(mesh_function_space, axes=axs[row, 1])
    axs[row, 1].set_title(
        f"{dataset_name} - {name} (torch finite diff)", fontsize=font_size
    )
    plt.colorbar(plt_obj3)

    # model output mesh
    fd.triplot(model_mesh, axes=axs[row, 2])
    axs[row, 2].set_title(f"{dataset_name} - Model output mesh", fontsize=font_size)

    # Sample values from firedrake
    mesh_function_space.dat.data[:] = fd_val.reshape(-1)[:]
    plt_obj3 = fd.tripcolor(mesh_function_space, axes=axs[row, 3])
    axs[row, 3].set_title(f"{dataset_name} - {name} (firedrake)", fontsize=font_size)
    plt.colorbar(plt_obj3)

    # MA output mesh
    fd_mesh.coordinates.dat.data[:] = fd_coord[:]
    fd.triplot(fd_mesh, axes=axs[row, 4])
    axs[row, 4].set_title(f"{dataset_name} - MA output mesh", fontsize=font_size)

    # Error map
    mesh_function_space.dat.data[:] = np.abs(
        fd_val.reshape(-1)[:] - model_val.reshape(-1)[:]
    )
    plt_obj3 = fd.tripcolor(mesh_function_space, axes=axs[row, 5])
    axs[row, 5].set_title(f"{dataset_name} - {name} (error L1)", fontsize=font_size)
    plt.colorbar(plt_obj3)
    row += 1

fig.savefig(
    f"{dataset_name}_output_comparison_{start_num+num_selected}_new_query_fd.png"
)
