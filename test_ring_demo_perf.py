import os
import time
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import pickle
import warpmesh as wm

print("Setting up solver.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_gpu_usage_to_file(filename):
    with open(filename, "w") as file:
        if torch.cuda.is_available():
            # Number of GPUs available
            num_gpus = torch.cuda.device_count()
            file.write(f"Number of GPUs available: {num_gpus}\n\n")

            for i in range(num_gpus):
                file.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
                file.write(
                    f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 2):.2f} MB\n"
                )
                file.write(
                    f"  Memory Cached: {torch.cuda.memory_reserved(i) / (1024 ** 2):.2f} MB\n"
                )
                file.write(
                    f"  Memory Free: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 2):.2f} MB\n"
                )
                file.write("\n")
        else:
            file.write(
                "CUDA is not available. Please check your PyTorch installation.\n"
            )


print("!!!!!device!!!!!! ", device)
#################### Load trained model ####################

with open("./pretrain_model/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)

config = SimpleNamespace(**config_data)
config.mesh_feat = ["coord", "monitor_val"]

print(config)

model = wm.M2N_T(
    deform_in_c=config.num_deform_in,
    gfe_in_c=config.num_gfe_in,
    lfe_in_c=config.num_lfe_in,
)
model_file_path = "./pretrain_model/model_999.pth"
model = wm.load_model(model, model_file_path)
model.eval()
model = model.to(device)
###########################################################

model_results = "./ring_demo_data/ring_ref_results.pkl"
input_sample_path = "./ring_demo_data/input_sample_data.pkl"
mesh_path = "./ring_demo_data/ring_demo_mesh.msh"
demo_output_path = "./ring_demo_output"
os.makedirs(demo_output_path, exist_ok=True)

with open(model_results, "rb") as f:
    plot_data_dict_model = pickle.load(f)
print(plot_data_dict_model)

with open(input_sample_path, "rb") as f:
    input_sample_data = pickle.load(f)
print(input_sample_data)

sample = input_sample_data.to(device)
total_infer_time = 0.0
all_infer_time = []
num_run = 20
with torch.no_grad():
    for _ in range(num_run):
        start_time = time.perf_counter()
        adapted_coord = model(sample)
        end_time = time.perf_counter()
        curr_infer_time = (end_time - start_time) * 1e3
        all_infer_time.append(curr_infer_time)
        total_infer_time += curr_infer_time
averaged_time = total_infer_time / num_run
print(
    f"Total model inference time: {total_infer_time} ms, averaged time: {averaged_time}"
)

# Check result
reference_adapted_mesh = plot_data_dict_model["mesh_model"]
adapted_coord_np = adapted_coord.cpu().detach().numpy()
assert np.allclose(
    adapted_coord_np, reference_adapted_mesh, rtol=1e-05, atol=1e-08
), "Model output mesh is not consistent to the reference"
print("Output is consistent to the reference.")

output_file = f"{demo_output_path}/test_ring_demo_perf_out.txt"
print(all_infer_time)
with open(output_file, "w") as f:
    f.write(", ".join([str(v) for v in all_infer_time]))
    f.write("\n")
    f.write("average time: " + str(averaged_time) + "\n")
    f.write("total time: " + str(total_infer_time) + "\n")
    f.write("num of vertices: " + str(adapted_coord_np.shape) + "\n")
    f.write("num of elements: " + str(sample.face.shape) + "\n")
print(f"write results to {output_file}.")

# Specify the output file name
output_file_gpu_info = f"{demo_output_path}/gpu_usage.txt"
dump_gpu_usage_to_file(output_file_gpu_info)
print(f"GPU usage information has been written to {output_file_gpu_info}.")

rows = 3
cols = 2
cmap = "seismic"

fig, ax = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10), layout="compressed")

## Firedrake visualization part
import firedrake as fd

mesh_og = fd.Mesh(mesh_path)
mesh_refer = fd.Mesh(mesh_path)
mesh_model = fd.Mesh(mesh_path)


og_function_space = fd.FunctionSpace(mesh_og, "CG", 1)
model_function_space = fd.FunctionSpace(mesh_model, "CG", 1)
mesh_refer_function_space = fd.FunctionSpace(mesh_refer, "CG", 1)

u_og = fd.Function(fd.FunctionSpace(mesh_og, "CG", 1))
u_ma = fd.Function(fd.FunctionSpace(mesh_refer, "CG", 1))
u_model = fd.Function(fd.FunctionSpace(mesh_model, "CG", 1))
monitor_values = fd.Function(og_function_space)

u_og_data = plot_data_dict_model["u_original"]
u_og.dat.data[:] = u_og_data

rows = 1
cols = 4
cmap = "seismic"
FONT_SIZE = 24

fig, ax = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10), layout="compressed")

fd.triplot(mesh_og, axes=ax[0])
ax[0].set_title("Original Mesh", fontsize=FONT_SIZE)
fd.tripcolor(u_og, axes=ax[1], cmap=cmap)
ax[1].set_title("Solution", fontsize=FONT_SIZE)

# Adapted mesh
mesh_model.coordinates.dat.data[:] = adapted_coord_np
fd.triplot(mesh_model, axes=ax[2])
ax[2].set_title("Adapated Mesh (UM2N)", fontsize=FONT_SIZE)

mesh_refer.coordinates.dat.data[:] = plot_data_dict_model["mesh_model"]
fd.triplot(mesh_model, axes=ax[3])
ax[3].set_title("Adapated Mesh (UM2N) Reference", fontsize=FONT_SIZE)


plt.savefig(f"{demo_output_path}/test_ring_demo_perf.png")
plt.show()
