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

yellow_log "Please deactivate conda enviroment before running this script, otherwise it will fail."

INSTALL_DIR=$(pwd)

# Download & install firedrake
cd ./install

# If previous file exists
if [ -d ./firedrake ]; then
  green_log "The directory has existed, nothing to be done"
fi

if [ ! -d ./firedrake ]; then
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python3 firedrake-install
  green_log "Firedrake has been installed"
fi

# Activate Firedrake enviroment
. ./firedrake/bin/activate

# Install PyTorch
green_log "Downloading PyTorch..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch3d
green_log "Downloading PyTorch3d..."
python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d"

# Install Movement
green_log "Downloading Movement..."
cd ${VIRTUAL_ENV}/src
git clone https://github.com/mesh-adaptation/movement.git
cd movement
pip install -e .

# Install WarpMesh
cd ${INSTALL_DIR}
pip install -e .
