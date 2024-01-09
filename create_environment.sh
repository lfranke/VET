#!/bin/bash

#git submodule update --init --recursive --jobs 0


source $(conda info --base)/etc/profile.d/conda.sh

conda update -n base -c defaults conda

conda create -y -n vet python=3.9.7
conda activate vet

conda install -y ncurses=6.3 -c conda-forge
conda install -y cuda -c nvidia/label/cuda-11.8.0
conda install -y configargparse=1.4 astunparse=1.6.3 numpy=1.21.2 ninja=1.10.2 pyyaml mkl=2022.0.1 mkl-include=2022.0.1 setuptools=58.0.4  cffi=1.15.0 typing_extensions=4.1.1 future=0.18.2 six=1.16.0 requests=2.27.1 dataclasses=0.8 tqdm
conda install -y -c conda-forge jpeg=9e
conda install -y -c conda-forge cmake=3.26.1 coin-or-cbc=2.10.5 glog=0.5.0 gflags=2.2.2 protobuf=3.13.0.1 freeimage tensorboard=2.8.0

conda install -y -c conda-forge cudnn=8.9.2

#./install_pytorch_precompiled.sh
