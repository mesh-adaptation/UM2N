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

run_id_collections = {"MRT":['mfn1hnrg'], "MRT-Sampling":['j9rjsxl1'], "MRN-GTE":['ywtfui2q'], "MRN-LTE":['uu515eu1'], "MRN":['0iwpdpnr'], "M2T":['gboubixk'], "M2N":['xqa8fnoj']}
test_ms = 25

models_to_compare = ["MRT", "MRT-Sampling", "MRN", "M2T", "M2N"]
# test dataset, for benchmarking loss effects on model performance
# test_dir = f"./data/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"
test_dir = f"./data/with_sampling/helmholtz/z=<0,1>_ndist=None_max_dist=6_<{test_ms}x{test_ms}>_n=100_aniso_full/data"
random_seed = 42

out_mesh_collections = {}
out_loss_collections = {}
for model_name in models_to_compare:
    run_id = run_id_collections[model_name][0]
    entity = 'mz-team' 
    project_name = 'warpmesh'

    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}")
    config = SimpleNamespace(**run.config)
    print(config)
    config.num_transformer_in = 4
    config.num_transformer_out = 16
    config.num_transformer_embed_dim = 64
    config.num_transformer_heads = 4
    config.num_transformer_layers = 1

    model = None
    if (config.model_used == "M2N"):
      model = wm.M2N(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
      )
    elif (config.model_used == "M2NAtten"):
      model = wm.M2NAtten(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
      )
    elif (config.model_used == "MRN"):
      model = wm.MRN(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
      )
    elif (config.model_used == "M2N_dynamic_drop"):
      model = wm.M2N_dynamic_drop(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
      )
    elif (config.model_used == "M2N_dynamic_no_drop"):
      model = wm.M2N_dynamic_no_drop(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
    elif (config.model_used == "MRNAtten"):
      model = wm.MRNAtten(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
    elif (config.model_used == "MRNGlobalTransformerEncoder"):
      model = wm.MRNGlobalTransformerEncoder(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
    elif (config.model_used == "MRNLocalTransformerEncoder"):
      model = wm.MRNLocalTransformerEncoder(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
        num_loop=config.num_deformer_loop,
    )
    elif (config.model_used == "MRTransformer"):
      model = wm.MRTransformer(
        num_transformer_in=config.num_transformer_in, 
        num_transformer_out=config.num_transformer_out, 
        num_transformer_embed_dim=config.num_transformer_embed_dim, 
        num_transformer_heads=config.num_transformer_heads, 
        num_transformer_layers=config.num_transformer_layers,
        deform_in_c=config.num_deform_in,
        num_loop=config.num_deformer_loop,
    )
    elif (config.model_used == "M2Transformer"):
      model = wm.M2Transformer(
        deform_in_c=config.num_deform_in,
        gfe_in_c=config.num_gfe_in,
        lfe_in_c=config.num_lfe_in,
    )
    else:
        raise Exception(f"Model {config.model_used} not implemented.")


    test_set = wm.MeshDataset(
        test_dir,
        transform=wm.normalise if wm.normalise else None,
        x_feature=config.x_feat,
        mesh_feature=config.mesh_feat,
        conv_feature=config.conv_feat,
        conv_feature_fix=config.conv_feat_fix,
        use_cluster=config.use_cluster,
    )

    loader = DataLoader(test_set, batch_size=1, shuffle=True)


    for file in run.files():
        print(file.name)

    epoch = 999
    target_file_name = "model_{}.pth".format(epoch)

    model_file = None
    for file in run.files():
        if file.name.endswith(target_file_name):
            model_file = file.download(replace=True)

    if model_file is None:
      print("No model file found on wandb! Load the local backup.")
      model_file = f"./out/{config.experiment_name}/{target_file_name}"
      target_file_name = model_file
    assert model_file is not None, "Model file not found either on wandb or local."
    print(target_file_name)
    model = wm.load_model(model, target_file_name)
    print(model)

    loss_func = torch.nn.L1Loss()
    model.to(device)
    model.eval()
    # with torch.no_grad():
    #   for i in range(10):
    #       idx = i
    #       sample = test_set[idx]
    #       print(sample)
    #       out = model(sample)
    #       print(f"{i} loss: {loss_func(out, sample.y)*1000}")

    #       compare_fig = wm.plot_mesh_compare(
    #           out.detach().cpu().numpy(), sample.y,
    #           sample.face
    #       )
    #       compare_fig.savefig(f"./out_images/img_method_{config.model_used}_reso_{test_ms}_{i}.png")


    out_mesh_collections[model_name] = []
    out_loss_collections[model_name] = []
    target_mesh = []
    target_face = []
    with torch.no_grad():
      cnt = 0
      torch.manual_seed(random_seed)
      for batch in loader:
          sample = batch.to(device)
          out = model(sample)
          deform_loss = loss_func(out, sample.y)*1000
          print(f"{model_name} {cnt} deform loss: {deform_loss}")
          out_mesh_collections[model_name].append(out.detach().cpu().numpy())
          out_loss_collections[model_name].append(deform_loss)
          target_mesh.append(sample.y.detach().cpu().numpy())
          target_face.append(sample.face.detach().cpu().numpy())
          # compare_fig = wm.plot_mesh_compare(
          #     out.detach().cpu().numpy(), sample.y,
          #     sample.face
          # )
          # compare_fig.savefig(f"./out_images/img_method_{config.model_used}_reso_{test_ms}_{cnt}.png")
          cnt += 1
          if cnt == 5:
            break


compare_fig = wm.plot_multiple_mesh_compare(out_mesh_collections, out_loss_collections, target_mesh, target_face)
compare_fig.tight_layout()
compare_fig.subplots_adjust(top=0.95)
compare_fig.suptitle(f"Ouput Mesh Comparsion (mesh resolution {test_ms}, dataloder seed: {random_seed})", fontsize=24)
compare_fig.savefig(f"./out_images/comparison_reso_{test_ms}_seed_{random_seed}.png")