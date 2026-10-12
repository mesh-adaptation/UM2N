# Towards Universal Mesh Movement Networks

This is the official code repository for the NeurIPS 2024 Spotlight paper [Toward Universal Mesh Movement Networks](https://arxiv.org/abs/2407.00382). 

Additional information: [[Project page]](https://erizmr.github.io/UM2N/)

<p align="center">
  <img src="./assets/UM2N_video_gif.gif" width="50%" height="50%" /><br>
  <em>UM2N(Universal Mesh Movement Networks) performs mesh adaptation for different PDEs and scenarios without re-training.</em>
</p>


## 🔎 Abstract

Solving complex Partial Differential Equations (PDEs) accurately and efficiently is an essential and challenging problem in all scientific and engineering disciplines. Mesh movement methods provide the capability to improve the accuracy of the numerical solution without increasing the overall mesh degree of freedom count. Conventional sophisticated mesh movement methods are extremely expensive and struggle to handle scenarios with complex boundary geometries. However, existing learning-based methods require re-training from scratch given a different PDE type or boundary geometry, which limits their applicability, and also often suffer from robustness issues in the form of inverted elements. In this paper, we introduce the Universal Mesh Movement Network (UM2N), which -- once trained -- can be applied in a non-intrusive, zero-shot manner to move meshes with different size distributions and structures, for solvers applicable to different PDE types and boundary geometries. UM2N consists of a Graph Transformer (GT) encoder for extracting features and a Graph Attention Network (GAT) based decoder for moving the mesh. We evaluate our method on advection and Navier-Stokes based examples, as well as a real-world tsunami simulation case. Our method outperforms existing learning-based mesh movement methods in terms of the benchmarks described above. In comparison to the conventional sophisticated Monge-Ampère PDE-solver based method, our approach not only significantly accelerates mesh movement, but also proves effective in scenarios where the conventional method fails.


The latest test status:

[![UM2N](https://github.com/mesh-adaptation/UM2N/actions/workflows/test_suite.yml/badge.svg)](https://github.com/mesh-adaptation/UM2N/actions/workflows/test_suite.yml)


## 🛠️ Installation

### All-in-one script

Just navigate to **project root** folder, open terminal and execute the
`install.sh` shell script:
``` shell
./install.sh
```
This will install [Firedrake](https://www.firedrakeproject.org/download.html)
and [Movement](https://github.com/mesh-adaptation/movement) under the `install`
folder, as well as the `UM2N` package. Note that the pytorch installed is a cpu version.

- GPU (cuda) support

For gpu support, please execute the:
``` shell
 ./install_gpu.sh {CUDA_VERSION}
 # e.g. `install_gpu.sh 118` for a CUDA version 11.8.
 ```


### Step-by-step approach

1. The mesh generation relies on Firedrake, which is a Python package. To
   install Firedrake, please follow the instructions on
   [firedrakeproject.org](https://www.firedrakeproject.org/download.html).

2. Use the virtual environment provided by Firedrake to install the dependencies
   of this project. The virtual environment is located at
   `/path/to/firedrake/bin/activate`. To activate the virtual environment, run
   `source /path/to/firedrake/bin/activate`.

3. The movement of the mesh is implemented by
   [mesh-adaptation/movement](https://github.com/mesh-adaptation/movement).
   To install it in the Firedrake virtual environment, follow these
   [instructions](https://github.com/mesh-adaptation/mesh-adaptation-docs/wiki/Installation-Instructions).

4. Install PyTorch into the virtual environment by following the instructions
   on the [PyTorch webpage](https://pytorch.org/get-started/locally).

5. UM2N's Chamfer loss needs `torch_geometric`'s `knn`, which requires the
   compiled `torch-cluster` package — install it matched to your torch
   version from step 4:
   ```shell
   pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
     -f https://data.pyg.org/whl/torch-<TORCH_VERSION>.html
   ```
   (`<TORCH_VERSION>` e.g. `2.8.0+cu129` — see [PyG's install
   docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-via-pre-built-wheels).)


6. Run `pip install .` in the root directory of this project to install the
   package and its other dependencies.


## 💿 Dataset generation (This is outdated)

In case you do not wish to generate the dataset by yourself, here is a
pre-generated dataset on Google Drive:
[link](https://drive.google.com/drive/folders/1sQ-9zWbTryCXwihqaqazrQ4Vp1MRdBPK?usp=sharing).
In this folder you can find all cases used to train/test the model. The naming
convention of the file is 'z=<0,1>_n_dist={number_of_distribution_used}_max_dist={maximum_distribution_used}_<{number_of_grid_in_x_direction}_{number_of_grid_in_y_direction}>_n={number_of_samples}_{data_set_type}'

If `n_dist = None`, then the number of Gaussian distribution used will be
randomly chosen from 1 to `max_dist`, otherwise, `n_dist` will be used to
generate a fixed number of Gaussian distribution version dataset.

The {data_set_type} will be either `'smpl'` or `'cmplx'`, indicating whether the
dataset is isotropic or anisotropic.

After download, you should put the downloaded folder `helmholtz` under
`data/dataset` folder.

### Generate the dataset by yourself (This is outdated)

```{shell}
. script/make_dataset.sh
```
This command will make following datasets by solving Monge-Ampère equation with
the following PDEs:

+ Burgers equation (on square domain)
+ Helmholtz equation (both square/random polygon domain)
+ Poisson equation (both square/random polygon domain)

User can modify the variables
```
n_dist_start=1
n_dist_end=10
n_grid_start=15
n_grid_end=35
```
defined in `script/make_dataset.sh` to generate datasets of different sizes.

The number of samples in the dataset can be changed by modifying the variable
`n_sample` in `script/build_helmholtz_dataset`.

## 🚀 Train the model (This is outdated)

A training notebook is provided: `script/train_um2n.ipynb`. Further training
details can be found in the notebook.

Here is also a link to pre-trained models:
[link](https://drive.google.com/drive/folders/1P_JMpU1qmLdmbGTz8fL5VO-lEBoP3_2n?usp=sharing)

## 📊 Evaluate the model (This is outdated)

There are a set of visualisation script under `script/` folder. The script can
be used to evaluate the model performance.

**Bear in mind that the path to datasets/model_weight in those files need
calibration**

## 📖 Documentation
The documentation is generated by Sphinx. To build the documentation, under the
`docs` folder.


## 🧩 Project Layout

```
├── UM2N (Implementation of the project)
│   ├── __init__.py
│   ├── generator (Dataset generator)
│   ├── processor (Data processor)
│   ├── helper (Helper functions)
│   ├── loader (Customized dataset and dataloader)
│   ├── model (MRN and M2N model implementation)
│   └── test (Simple tests for the model)
├── data (Datasets are generated here)
│   ├── dataset
│   └── output
├── docs (Documentation)
│   ├── conf.py
│   └── index.rst
├── play
│   ├── play_conv_feat.py
│   └── play_dataset.py
├── script (Utility scripts)
│   ├── make_dataset.sh (Script for making datasets of different sizes)
│   ├── build_helmholtz_dataset.py (Build helmholtz dataset)
│   ├── compare.py (Compare the performance of different models)
│   ├── evaluate.py 
│   ├── gradual_change.py
│   ├── plot.py
│   ├── train_model.py
│   └── ...
├── test
│   ├── test_import.py
│   └── ...
├── install.sh (Installation script for UM2N and its dependencies)
├── pyproject.toml (Top-level metadata for Python project)
└── README.md (Project summary and useful information)
```

