import glob
import pandas as pd
import matplotlib.pyplot as plt

log_files = glob.glob("./eval/2024_01_24_00_17_49_MRTransformer_999_8ndi2teh/helmholtz_square/log/*.csv")
# print(log_files)
error_MA = 0
error_model = 0
error_og = 0
time_MA = 0
num_tangle = 0
time_model = 0
pass_count = 0
fail_count = 0
total_count = 0

error_reduction_MA = []
error_reduction_model = []
for file_names in log_files:
    total_count += 1
    log_df = pd.read_csv(file_names)
    # print(log_df['tangled_element'][0], log_df['tangled_element'][0] == 0)
    if log_df['tangled_element'][0] == 0:
        pass_count += 1
        error_og += log_df['error_og'][0]
        error_MA += log_df['error_ma'][0]
        error_model += log_df['error_model'][0]
        error_reduction_MA.append(log_df['error_reduction_MA'])
        error_reduction_model.append(log_df['error_reduction_model'])
        time_MA += log_df['time_consumption_MA'][0]
        time_model += log_df['time_consumption_model'][0]
    else:
        fail_count += 1
        num_tangle += log_df['tangled_element'][0]
print(f"passed num: {pass_count}, failed num: {fail_count}")
plt.plot([x for x in range(len(error_reduction_MA))], error_reduction_MA, label='error_reduction_MA')
plt.plot([x for x in range(len(error_reduction_model))], error_reduction_model, label='error_reduction_model')
plt.legend()

plt.savefig("./vis_summary.png")
plt.show()