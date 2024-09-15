import pickle

model_names = ["M2N", "M2N", "MRTransformer", "M2T"]
run_ids = ["cyzk2mna", "u4uxcz1e", "99zrohiu", "gywsmly9"]
run_id_model_mapping = {
    "cyzk2mna": "M2N",
    "u4uxcz1e": "M2N-en",
    "99zrohiu": "MRN",
    "gywsmly9": "M2T-w-edge",
}
trained_epoch = 999
problem_type = "helmholtz_square"

# dataset_path = "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_6"
dataset_path = "./data/dataset_meshtype_6/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=100_aniso_full_meshtype_6"
# dataset_path = "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.05_n=100_aniso_full_meshtype_2"
# dataset_path = "./data/dataset_meshtype_2/helmholtz/z=<0,1>_ndist=None_max_dist=6_lc=0.028_n=100_aniso_full_meshtype_2"

dataset_name = dataset_path.split("/")[-1]
result_folder = f"./compare_output/{dataset_name}"

ret_file = f"{result_folder}/ret_stat.pkl"
with open(ret_file, "rb") as f:
    ret_dict = pickle.load(f)
# print(ret_dict)

# Compare between error reduction
error_reduction_avg = []


err_reduction_avg = ret_dict["ma"]["error_reduction_avg"]
err_reduction_std = ret_dict["ma"]["error_reduction_std"]
err_reduction_sum_avg = ret_dict["ma"]["error_reduction_sum_avg"]
print(
    f"[MA] error reduction avg {err_reduction_avg:.4f} std: {err_reduction_std:.4f}, err reduction sum avg: {err_reduction_sum_avg:.4f}"
)

for run_id in run_ids:
    name = ret_dict[run_id]["name"]
    tangled_num = ret_dict[run_id]["tangled_num"]
    err_reduction_avg = ret_dict[run_id]["error_reduction_avg"]
    err_reduction_std = ret_dict[run_id]["error_reduction_std"]
    err_reduction_sum_avg = ret_dict[run_id]["error_reduction_sum_avg"]
    print(
        f"[{name}] error reduction avg {err_reduction_avg:.4f} std: {err_reduction_std:.4f}, err reduction sum avg: {err_reduction_sum_avg:.4f} tangled num: {tangled_num}"
    )
    # if name == "MRN":
    #     print(ret_dict[run_id]["error_reduction"])
    #     print(ret_dict[run_id]["error"])
