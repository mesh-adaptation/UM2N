import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def write_stat(eval_dir, log_folder_name="log"):
    log_dir = os.path.join(eval_dir, log_folder_name)
    file_path = os.path.join(log_dir, "log*.csv")
    log_files = glob.glob(file_path)
    log_files = sorted(log_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))  # noqa
    dfs = [pd.read_csv(log_file) for log_file in log_files]
    df = pd.concat(dfs)
    df.to_csv(os.path.join(eval_dir, "df.csv"))

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    axs[0, 0].set_title("PDE Error on MA Mesh")
    sns.histplot(df["error_ma"], kde=True, ax=axs[0, 0], bins=30)
    axs[0, 0].set_xlabel("PDE Error")
    axs[0, 0].set_ylabel("Frequency")
    fig.savefig(os.path.join(eval_dir, "df.png"))

    axs[0, 1].set_title("PDE Error on Model Mesh")
    data_to_plot = df[df["error_model"] != -1]["error_model"]
    sns.histplot(data_to_plot, kde=True, ax=axs[0, 1], bins=30)
    axs[0, 1].set_xlabel("PDE Error")
    axs[0, 1].set_ylabel("Frequency")

    axs[0, 2].set_title("PDE Error Compare")
    data_to_plot = df[df["error_model"] != -1]["error_model"]
    sns.histplot(df["error_ma"], kde=True, ax=axs[0, 2], bins=30, label="MA")
    sns.histplot(data_to_plot, kde=True, ax=axs[0, 2], bins=30, label="Model")
    axs[0, 2].legend()
    axs[0, 2].set_xlabel("PDE Error")
    axs[0, 2].set_ylabel("Frequency")

    axs[1, 0].set_title("PDE Error Reduction Rate on MA Mesh")
    data_to_plot = df[df["error_model"] != -1]["error_reduction_model"]
    sns.histplot(df["error_reduction_MA"], kde=True, ax=axs[1, 0], bins=30, label="MA")  # noqa
    axs[1, 0].legend()
    axs[1, 0].set_xlabel("PDE Error Reduction Rate (%)")
    axs[1, 0].set_ylabel("Frequency")

    axs[1, 1].set_title("PDE Error Reduction Rate on Model Mesh")
    data_to_plot = df[df["error_model"] != -1]["error_reduction_model"]
    sns.histplot(data_to_plot, kde=True, ax=axs[1, 1], bins=30, label="Model")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("PDE Error Reduction Rate (%)")
    axs[1, 1].set_ylabel("Frequency")

    axs[1, 2].set_title("PDE Error Reduction Rate Compare")
    data_to_plot = df[df["error_model"] != -1]["error_reduction_model"]
    sns.histplot(df["error_reduction_MA"], kde=True, ax=axs[1, 2], bins=30, label="MA")  # noqa
    sns.histplot(data_to_plot, kde=True, ax=axs[1, 2], bins=30, label="Model")
    axs[1, 2].legend()
    axs[1, 2].set_xlabel("PDE Error Reduction Rate (%)")
    axs[1, 2].set_ylabel("Frequency")

    return {
        "fig": fig,
        "df": df,
    }


if __name__ == "__main__":
    path = "/Users/chunyang/projects/WarpMesh/eval/MRN_1499_gwts42h7/poisson_poly/z=<0,1>_ndist=None_max_dist=6_lc=0.04_n=400_aniso_full_meshtype_6/2024_01_31_13_18_58"  # noqa
    write_stat(path)
