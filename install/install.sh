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

# Download mesh-adaptation/movement package.
green_log "Start downloading Movement..."

if [ -d ./install/movement ]; then
  yellow_log "Movement dir exists, abort downloading"
fi

if [ ! -d ./install/movement ]; then
  yellow_log "Movement does not exist, start downloading Movement"
  git submodule add https://github.com/mesh-adaptation/movement.git install/movement
fi

# Download & install firedrake
cd ./install

# If previous file exists
if [ -d ./firedrake ]; then
  green_log "The directory has existed, nothing to be done"
fi

if [ ! -d ./firedrake ]; then
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
  python3 firedrake-install --disable-ssh
  green_log "Firedrake has been installed"
fi

# Activate Firedrake enviroment
. ./firedrake/bin/activate

# Install Movement
cd movement
pip install -e .

# Install WarpMesh
cd ../..
pip install -r requirements.txt
pip install -e .
