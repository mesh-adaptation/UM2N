import warpmesh as wm
import wandb
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
import torch
import movement as mv
import matplotlib.tri as tri
from torch_geometric.data import DataLoader
from types import SimpleNamespace
from io import BytesIO

import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# run_id = 'welbby7t'
# run_id = 'vwopbol5'
# run_id = 'ixtqouzl'
# run_id = '0iwpdpnr' # MRN
run_id = 'mfn1hnrg' # MRT
run_id = 'uu515eu1' # MRN-LTE
run_id = 'ywtfui2q' # MRN-GTE
run_id = 'gboubixk' # M2T
run_id = 'xqa8fnoj' # M2N
# run_id = 'l9cfh1wj' # MRT
# run_id = 'j9rjsxl1' # MRT + sampling
# run_id = 'hegubzg0' # MRN + sampling
run_id = '2b0ouh5p' # MRT + no up bottom left right

run_id = 'kg4y9zak' # MRT + mask 0.95
run_id = 'lmcata0v' # MRT + mask 0.75
run_id = '8becmygf' # MRT + mask 0.50
run_id = 'n2qcocej' # MRT + mask 0.25

run_id = 'esn5ynfq' # MRT + 1 layer recurrent attention mask 0.50
run_id = '7yaerq40' # MRT + 1 layer recurrent attention mask 0.50~0.90

run_id = 'lvcal7vq' # MRT + 1 layer recurrent + mask 0.5
run_id = 't233y3ik' # MRT + 1 layer recurrent + mask 0.75
run_id = 'yl8fmiip' # MRT + 1 layer recurrent + mask 0.95
run_id = 'nevv2a0d' # MRT + 1 layer recurrent + mask 0.5 ~ 0.9

run_id = 'kx25grpm' # MRT + 3 layers recurrent
run_id = '790xybc1' # MRT + 2 layers recurrent
run_id = 'zdj9ocmw' # MRT + 1 layer recurrent

run_id = '2wta7yed' # MRT-1R with sampling
run_id = '0bsy6m45' # MRT-1R no hessian


run_id = 'fzgaycnv' # MRT-1R output phi

run_id = 'mug27xhl' # MRT-1R output phi convex loss


run_id = '9ygg08yg' # MRT-1R output phi, constrain bd

run_id = 'u14bt77h' # output phi grad

run_id = 'f4q1v2pd' # output coord

run_id = 'kst5ig88' # output phi grad large eq residual


run_id = 'c2kyy4vl' # purely unsupervised
run_id = 'a2af7x3j' # weight_d = 0.01 weight_u


run_id = 'bzlj9vcl' # unsupervised 1 1 1

run_id = '8ndi2teh' # unsupervised 1 1 1, small unsupervised


run_id = '2f4florr' # unsupervised 1 01 01,  large unsupervised

run_id = 'fhmqq8eg' # supervised phi grad test 


run_id_collections = {"MRT":['mfn1hnrg'],

                      "MRT-no-udlr":['2b0ouh5p'],
                      "MRT-1R-phi":['fzgaycnv'],
                      "MRT-1R-phi-convex":['mug27xhl'],
                      "MRT-1R-phi-bd":['9ygg08yg'],
                      "MRT-1R-phi-grad":['u14bt77h'],
                      "MRT-1R-phi-grad-eq":['kst5ig88'],
                      "MRT-1R-coord":['f4q1v2pd'],
                      "MRT-1R-phi-grad-un":['c2kyy4vl'],
                      "MRT-1R-phi-grad-quasi-un":['a2af7x3j'],
                      "MRT-1R-phi-grad-un-111": ['bzlj9vcl'],
                      "MRT-1R-phi-grad-un-111-small": ['8ndi2teh'],
                      "MRT-1R-phi-grad-un-111-large": ['2f4florr'],
                      "MRT-1R-phi-grad-test": ['fhmqq8eg'],

                      "MRT-1R":['zdj9ocmw'],
                      "MRT-2R":['790xybc1'],
                      "MRT-3R":['kx25grpm'],
                      "MRT-1R-sampling":['2wta7yed'],
                      "MRT-1R-no-hessian":['0bsy6m45'],

                      "MRT-1R-atten-mask0.50":['esn5ynfq'],
                      "MRT-1R-atten-mask0.50~0.90":['7yaerq40'],
                      
                      "MRT-1R-mask0.95":['yl8fmiip'],
                      "MRT-1R-mask0.75":['t233y3ik'],
                      "MRT-1R-mask0.50":['lvcal7vq'],
                      "MRT-1R-mask0.50~0.9":['nevv2a0d'],

                      "MRT-mask0.95":['kg4y9zak'],
                      "MRT-mask0.75":['lmcata0v'],
                      "MRT-mask0.50":['8becmygf'],

                      "MRT-mask0.25":['n2qcocej'],
                      "MRT-Sampling":['j9rjsxl1'], 
                      "MRN-Sampling":['hegubzg0'], 
                      "MRN-GTE":['ywtfui2q'], 
                      "MRN-LTE":['uu515eu1'], 
                      "MRN":['0iwpdpnr'], 
                      "M2T":['gboubixk'], 
                      "M2N":['xqa8fnoj']}

dataset_name = 'helmholtz'
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
models_to_compare = ["MRT-1R-phi-grad-test"]
# models_to_compare = ["MRT-1R", "MRT-1R-no-hessian"]
# test dataset, for benchmarking loss effects on model performance


if dataset_name == 'helmholtz':
  # test_dir = f"./data/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"

  # dataset_dir = f"./data/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=10_aniso_full"
  dataset_dir = f"./data/dataset/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full"
  # test_dir = f"./data/with_sampling/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"
  # test_dir = f"./data/large_scale_test/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"
  # test_dir = f"./data/helmholtz_poly/helmholtz_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.06_n=400_aniso_full/data"
elif dataset_name == 'swirl':
  dataset_dir = f"./data/dataset/swirl/sigma_0.017_alpha_1.0_r0_0.2_lc_0.05_interval_5"
  # Swirl
  # test_dir = f"./data/swirl/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=iso_pad/data"
test_dir = f"{dataset_dir}/data"

random_seed = 1


phi_MA_collections = {}
phix_MA_collections = {}
phiy_MA_collections = {}
out_mesh_collections = {}
out_phix_collections = {}
out_phiy_collections = {}
out_phixx_collections = {}
out_phiyy_collections = {}
out_loss_collections = {}
out_atten_collections = {}
for model_name in models_to_compare:
    run_id = run_id_collections[model_name][0]
    entity = 'mz-team' 
    project_name = 'warpmesh'

    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    print(config)
    config.num_transformer_in = 4
    if 'no-hessian' in model_name:
      config.num_transformer_in = 3
    config.num_transformer_out = 16
    config.num_transformer_embed_dim = 64
    config.num_transformer_heads = 4
    config.num_transformer_layers = 1
    if run_id == '9ygg08yg':
      config.deform_out_type = 'phi'
    
    model = None
    if config.model_used == "MRTransformer":
      model = wm.MRTransformer(
        num_transformer_in=config.num_transformer_in, 
        num_transformer_out=config.num_transformer_out, 
        num_transformer_embed_dim=config.num_transformer_embed_dim, 
        num_transformer_heads=config.num_transformer_heads, 
        num_transformer_layers=config.num_transformer_layers,
        deform_in_c=config.num_deform_in,
        deform_out_type=config.deform_out_type,
        num_loop=config.num_deformer_loop,
        device=device
    )
    else:
        raise Exception(f"Model {config.model_used} not implemented.")
    config.mesh_feat.extend(['phi', 'grad_phi'])
    print("mesh feat type ", config.mesh_feat)
    test_set = wm.MeshDataset(
        test_dir,
        # transform=wm.normalise if wm.normalise else None,
        transform=None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        use_cluster=config.use_cluster,
        load_jacobian=True
    )
    print("mesh feat mesh feat ", test_set[0].mesh_feat.shape)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    epoch = 999
    # TODO: the MRN-Sampling ('hegubzg0') only trained 800 epochs
    if run_id == 'hegubzg0':
       epoch = 799

    target_file_name = "model_{}.pth".format(epoch)

    model_file = None
    for file in run.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(root='.', replace=True)
            print('download file ', model_file)
    
    model_file = None
    if model_file is None:
      print("No model file found on wandb! Load the local backup.")
      model_file = f"./out/{config.experiment_name}/{target_file_name}"
      target_file_name = model_file
    assert model_file is not None, "Model file not found either on wandb or local."
    print(target_file_name)
    model = wm.load_model(model, model_file)
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
    out_phiyy_collections[model_name] = []

    phi_MA_collections[model_name] = []
    phix_MA_collections[model_name] = []
    phiy_MA_collections[model_name] = []

    target_mesh = []
    target_face = []
    target_hessian_norm = []
    num_step_recurrent = 5
    # with torch.no_grad():
    cnt = 0
    torch.manual_seed(random_seed)
    for batch in loader:
        sample = batch.to(device)
        print("mesh feat shape ", sample.mesh_feat.shape)
        if model_name == 'MRT':
          out = model.move(sample, num_step=5)
        else:
          if 'MRT-1R' in model_name:
            if 'phi' in model_name:
              phi_MA_collections[model_name].append(sample.mesh_feat[:, 4].detach().cpu().numpy())
              phix_MA_collections[model_name].append(sample.mesh_feat[:, 5].detach().cpu().numpy())
              phiy_MA_collections[model_name].append(sample.mesh_feat[:, 6].detach().cpu().numpy())

              # Mannually normlize for model input
              mesh_val_feat = sample.mesh_feat[:, 2:]  # value feature (no coord)
              min_val = torch.min(mesh_val_feat, dim=0).values
              max_val = torch.max(mesh_val_feat, dim=0).values
              max_abs_val = torch.max(torch.abs(min_val), torch.abs(max_val))
              sample.mesh_feat[:, 2:] = sample.mesh_feat[:, 2:] / max_abs_val


              sample.x.requires_grad = True
              sample.mesh_feat.requires_grad = True
              out, (phix, phiy) = model.move(sample, num_step=1)
              feat_dim = sample.mesh_feat.shape[-1]
              # mesh_feat [coord_x, coord_y, u, hessian_norm]
              node_num = sample.mesh_feat.reshape(1, -1, feat_dim).shape[1]

              out_phix_collections[model_name].append(phix.detach().cpu().numpy())
              out_phiy_collections[model_name].append(phiy.detach().cpu().numpy())
              
              hessian_seed = torch.ones(phix.shape).to(device)
              phix_grad = torch.autograd.grad(phix, sample.mesh_feat, grad_outputs=hessian_seed, retain_graph=True, create_graph=True, allow_unused=True)[0]
              phiy_grad = torch.autograd.grad(phiy, sample.mesh_feat, grad_outputs=hessian_seed, retain_graph=False, create_graph=False, allow_unused=True)[0]

              # print(f"phix grad: {phix_grad.shape}, phiy grad: {phiy_grad.shape}")
              phixx = phix_grad[:, 0]
              phixy = phix_grad[:, 1]
              phiyx = phiy_grad[:, 0]
              phiyy = phiy_grad[:, 1]

              out_phixx_collections[model_name].append(phixx.detach().cpu().numpy())
              out_phiyy_collections[model_name].append(phiyy.detach().cpu().numpy())



              # # Compute the residual to the equation
              # grad_seed = torch.ones(out.shape).to(device)
              # phi_grad = torch.autograd.grad(out, sample.x, grad_outputs=grad_seed, retain_graph=True, create_graph=True, allow_unused=True)[0]
              # phix = phi_grad[:, 0]
              # phiy = phi_grad[:, 1]

              # New coord
              # coord_x = (sample.x[:, 0] + phix).reshape(1, node_num, 1)
              # coord_y = (sample.x[:, 1] + phiy).reshape(1, node_num, 1)
              # out = torch.cat([coord_x, coord_y], dim=-1).reshape(-1, 2)
            elif 'MRT-1R-coord' in model_name:
              out, (phix, phiy) = model.move(sample, num_step=1)
            else:
              out = model.move(sample, num_step=1)
          elif 'MRT-2R' in model_name:
            out = model.move(sample, num_step=2)
          elif 'MRT-3R' in model_name:
            out = model.move(sample, num_step=3)
          else:
            out = model.move(sample, num_step=5)
        print(out.shape)
        if 'MRT' in model_name:
            attentions = model.get_attention_scores(sample)
        deform_loss = loss_func(out, sample.y)*1000
        print(f"{model_name} {cnt} deform loss: {deform_loss}, mesh vertices: {out.shape}")
        out_mesh_collections[model_name].append(out.detach().cpu().numpy())
        out_loss_collections[model_name].append(deform_loss)
        out_atten_collections[model_name].append(attentions)
        target_mesh.append(sample.y.detach().cpu().numpy())
        target_face.append(sample.face.detach().cpu().numpy())
        target_hessian_norm.append(sample.mesh_feat[:,-1].detach().cpu().numpy())
        # compare_fig = wm.plot_mesh_compare(
        #     out.detach().cpu().numpy(), sample.y,
        #     sample.face
        # )
        # compare_fig.savefig(f"./out_images/img_method_{config.model_used}_reso_{test_ms}_{cnt}.png")
        cnt += 1
        if cnt == num_sample_vis:
          break


# compare_fig = wm.plot_multiple_mesh_compare(out_mesh_collections, out_loss_collections, target_mesh, target_face)
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

model_name = "MRT-1R-phi-grad-test"
num_selected = 1
phix_grad = out_phix_collections[model_name][num_selected]
phiy_grad = out_phiy_collections[model_name][num_selected]

phixx = out_phixx_collections[model_name][num_selected]
phiyy = out_phiyy_collections[model_name][num_selected]

coord = out_mesh_collections[model_name][num_selected]


phi_sample = phi_MA_collections[model_name][num_selected]
phix_sample = phix_MA_collections[model_name][num_selected]
phiy_sample = phiy_MA_collections[model_name][num_selected]

import os

mesh_gen = wm.UnstructuredSquareMesh()
model_mesh = mesh_gen.load_mesh(file_path=os.path.join(f"{dataset_dir}/mesh", "mesh0.msh"))

# Phi grad x
# model_mesh = fd.UnitSquareMesh(test_ms, test_ms)
model_mesh.coordinates.dat.data[:] = coord[:]
det_model_raw = fd.Function(
    fd.FunctionSpace(model_mesh, "CG", 1))
det_model_raw.dat.data[:] = phix_grad.reshape(-1)[:]

fig, axs = plt.subplots(1, 3, figsize=(30, 8))

plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[0])
axs[0].set_title(f"{dataset_name} - Phi grad x (torch autograd)")
plt.colorbar(plt_obj3)


det_model_raw.dat.data[:] = phix_sample.reshape(-1)[:]
plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[1])
axs[1].set_title(f"{dataset_name} - Phi grad x (firedrake)")
plt.colorbar(plt_obj3)

fd.triplot(model_mesh, axes=axs[2])
fig.savefig(f"{dataset_name}_test_grad_x.png")


# Phi grad y
# model_mesh = fd.UnitSquareMesh(test_ms, test_ms)
model_mesh.coordinates.dat.data[:] = coord[:]
det_model_raw = fd.Function(
    fd.FunctionSpace(model_mesh, "CG", 1))
det_model_raw.dat.data[:] = phiy_grad.reshape(-1)[:]

fig, axs = plt.subplots(1, 3, figsize=(30, 8))

plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[0])
axs[0].set_title(f"{dataset_name} - Phi grad y (torch grad)")
plt.colorbar(plt_obj3)

det_model_raw.dat.data[:] = phiy_sample.reshape(-1)[:]
plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[1])
axs[1].set_title(f"{dataset_name} - Phi grad y (firedrake)")
plt.colorbar(plt_obj3)

fd.triplot(model_mesh, axes=axs[2])
fig.savefig(f"{dataset_name}_test_grad_y.png")


# Phi grad grad x
# model_mesh = fd.UnitSquareMesh(test_ms, test_ms)
model_mesh.coordinates.dat.data[:] = coord[:]
det_model_raw = fd.Function(
    fd.FunctionSpace(model_mesh, "CG", 1))
det_model_raw.dat.data[:] = phixx.reshape(-1)[:]

fig, axs = plt.subplots(1, 2, figsize=(20, 8))
axs[0].set_title(f"{dataset_name} - Phi grad x grad x")

plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[0])
plt.colorbar(plt_obj3)

fd.triplot(model_mesh, axes=axs[1])
fig.savefig(f"{dataset_name}_test_grad_xx.png")

# # Phi grad grad x (finite difference)
# model_mesh = fd.UnitSquareMesh(test_ms, test_ms)
# model_mesh.coordinates.dat.data[:] = coord[:]
# det_model_raw = fd.Function(
#     fd.FunctionSpace(model_mesh, "CG", 1))
# det_model_raw.dat.data[:] = np.diff(phix_grad, append=[0]).reshape(-1)[:]

# fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[0])
# axs[0].title(f"{dataset_name} - Phi grad x grad x finite difference")
# plt.colorbar(plt_obj3)

# fd.triplot(model_mesh, axes=axs[1])
# fig.savefig(f"{dataset_name}_test_grad_xx_finite_difference.png")


# Phi grad grad y
# model_mesh = fd.UnitSquareMesh(test_ms, test_ms)
model_mesh.coordinates.dat.data[:] = coord[:]
det_model_raw = fd.Function(
    fd.FunctionSpace(model_mesh, "CG", 1))
det_model_raw.dat.data[:] = phiyy.reshape(-1)[:]

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

plt_obj3 = fd.tripcolor(det_model_raw, axes=axs[0])
axs[0].set_title(f"{dataset_name} - Phi grad y grad y")
plt.colorbar(plt_obj3)

fd.triplot(model_mesh, axes=axs[1])
fig.savefig(f"{dataset_name}_test_grad_yy.png")
