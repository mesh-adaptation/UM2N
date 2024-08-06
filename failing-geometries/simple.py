import time
import torch
import yaml
import firedrake as fd
import numpy as np
import warpmesh as wm
from types import SimpleNamespace
from inference_utils import get_conv_feat, find_edges, find_bd, InputPack
import movement

print("Setting up solver.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('../pretrain_model/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

config = SimpleNamespace(**config_data)

# Append the monitor val at the end
# config.mesh_feat.append("monitor_val")
# config.mesh_feat = ["coord", "u", "monitor_val"]
config.mesh_feat = ["coord", "monitor_val"]

# print("# Evaluation Pipeline Started\n")
print(config)

model = wm.M2N_T(
    deform_in_c=config.num_deform_in,
    gfe_in_c=config.num_gfe_in,
    lfe_in_c=config.num_lfe_in,
)
model_file_path = "../pretrain_model/model_999.pth"
model = wm.load_model(model, model_file_path)
model.eval()
model = model.to(device)
###########################################################



mesh = fd.Mesh("double_basin.msh")
# mesh = fd.Mesh("square.msh")
# mesh = fd.Mesh("headland.msh")
Q = fd.FunctionSpace(mesh, "CG", 1)


def monitor(mesh):
    x, y = fd.SpatialCoordinate(mesh)
    # return fd.exp(-(x-0.5)**2/(0.1)**2)*4 + 1
    # return fd.conditional(fd.And(abs(x-0.5)<0.2, abs(y-0.5)<0.2), 5, 1)
    return 1. + 4*x #*fd.exp(-y**2/(.1)**2)


# Extract input features
coords = mesh.coordinates.dat.data_ro.copy()
print(f"coords {coords.shape}")
# print(f"conv feat {conv_feat.shape}")
edge_idx = find_edges(mesh, Q)
print(f"edge idx {edge_idx.shape}")
bd_mask, _, _, _, _ = find_bd(mesh, Q)
print(f"boundary mask {bd_mask.shape}")


u_list = []
step_cnt = 0
monitor_val = fd.Function(Q, name='monitor')
fout = fd.File('out.pvd')

with torch.no_grad():

    monitor_val.interpolate(monitor(mesh))
    filter_monitor_val = np.minimum(1e3, monitor_val.dat.data[:])
    filter_monitor_val = np.maximum(0, filter_monitor_val)
    monitor_val.dat.data[:] = filter_monitor_val / filter_monitor_val.max()
    fout.write(monitor_val)
    conv_feat = get_conv_feat(mesh, monitor_val)
    end_time = time.perf_counter()
    sample = InputPack(
            coord=coords,
            monitor_val=monitor_val.dat.data_ro.reshape(-1, 1),
            edge_index=edge_idx, bd_mask=bd_mask,
            conv_feat=conv_feat,
            stack_boundary=False)
    adapted_coord = model(sample)
    mesh.coordinates.dat.data[:] = adapted_coord.cpu().detach().numpy()
    fout.write(monitor_val)

# back to original mesh:
mesh.coordinates.dat.data[:] = coords
mover = movement.MongeAmpereMover(mesh, monitor, method='relaxation',
        rtol=1e-4, maxiter=500) #, fix_boundary_nodes=[4])
# for the headland case, this setup allows movement along all boundaries
# currently we can stop movement alltogether along all boundary with
# fix_boundary_nodes=True - but not fix it for one boundary only
# If you want to fix the nodes only along the top with the headland,
# change line 559 of movement/monge_ampere.py:
#     if self.fix_boundary_nodes:
# to:
#     if tag in self.fix_boundary_nodes
# and setup the solver with fix_boundary_nodes=[4]) i.e. you specify a list
# of ids of the boundaries that you want to fix, rather than True or False
del(mover.tangling_checker)  # skip tangling check, so we can look at the tangled output

try:
    mover.move()
except fd.ConvergenceError:
    print("FAILED TO CONVERGE!!!")

V = fd.FunctionSpace(mover.mesh, "CG", 1)

foo = fd.Function(V, name='monitor')
fout.write(foo)
