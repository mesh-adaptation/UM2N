#!/bin/bash

# This script will install necessary packages for the project

function yellow_log() {
  local DATE_N=$(date "+%Y-%m-%d %H:%M:%S")
  local color="\033[33m"
  echo -e "$DATE_N$color $*  \033[0m"
}

function green_log() {
  local DATE_N=$(date "+%Y-%m-%d %H:%M:%S")
  local color="\033[32m"
  echo -e "$DATE_N$color $*  \033[0m"
}

yellow_log "Please deactivate conda environment before running this script, otherwise it will fail."

# Check if CUDA version is provided as an argument
if [ -z "$1" ]; then
  echo "Error: No CUDA version provided."
  echo "Usage: ./install.sh <cuda_version>"
  exit 1
else
  CUDA_VERSION=$1
  green_log "Using CUDA version: $CUDA_VERSION"
fi

UM2N_ROOT=$(pwd)
INSTALL_DIR=${UM2N_ROOT}/install

# Download and install Firedrake
mkdir -p ${INSTALL_DIR}
cd ${INSTALL_DIR}

# Build Firedrake if it does not already exist
if [ -d ${INSTALL_DIR}/firedrake ]; then
  green_log "A Firedrake installation already exists, nothing to be done"
else
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python3 firedrake-install
  green_log "Firedrake has been installed"
fi

# Activate Firedrake environment
. ${INSTALL_DIR}/firedrake/bin/activate

# Install PyTorch
green_log "Downloading PyTorch..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# Install PyTorch3d
green_log "Downloading PyTorch3d..."
python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d"

# Install Movement
green_log "Downloading Movement..."
git clone https://github.com/mesh-adaptation/movement.git ${VIRTUAL_ENV}/src/movement
pip install -e ${VIRTUAL_ENV}/src/movement

# Install WarpMesh
pip install -e ${UM2N_ROOT}
