# VET: Visual Error Tomography for Point Cloud Completion and High-Quality Neural Rendering

<div style="text-align: center;">Linus Franke, Darius Rückert, Laura Fink, Matthias Innmann, Marc Stamminger</div>



**Abstract:** In the last few years, deep neural networks opened the doors for big advances in novel view synthesis.
Many of these approaches are based on a (coarse) proxy geometry obtained by structure from motion algorithms.
Small deficiencies in this proxy can be fixed by neural rendering, but larger holes or missing parts, as they commonly
appear for thin structures or for glossy regions, still lead to distracting artifacts and temporal instability.
In this paper, we present a novel neural-rendering-based approach to detect and fix such deficiencies.
As a proxy, we use a point cloud, which allows us to easily remove outlier geometry and to fill in missing geometry
without complicated topological operations.
Keys to our approach are (i) a differentiable, blending point-based renderer that can blend out redundant points, as
well as (ii) the concept of Visual Error Tomography (VET), which allows us to lift 2D error maps to identify 3D-regions
lacking geometry and to spawn novel points accordingly.
Furthermore, (iii) by adding points as nested environment maps, our approach allows us to generate high-quality
renderings of the surroundings in the same pipeline.
In our results, we show that our approach can improve the quality of a point cloud obtained by structure from motion and
thus increase novel view synthesis quality significantly.
In contrast to point growing techniques, the approach can also fix large-scale holes and missing thin structures
effectively.
Rendering quality outperforms state-of-the-art methods and temporal stability is significantly improved, while rendering
is possible at real-time frame rates.

[[Project Page]](https://lfranke.github.io/vet/) [[Paper]](https://arxiv.org/abs/2311.04634) [[Youtube]](https://youtu.be/adH6GyqC4Jk)

## Citation

```
@article{franke2023vet,
    title={VET: Visual Error Tomography for Point Cloud Completion and High-Quality Neural Rendering},
    author={Linus Franke and Darius R{\"u}ckert and Laura Fink and Matthias Innmann and Marc Stamminger},
    booktitle = {ACM SIGGRAPH Asia 2023 Conference Proceedings},
    year = {2023}
}

```

## Install Requirements

Supported Operating Systems: Ubuntu 22.04

Supported Compiler: g++-9

Software Requirement: Conda (Anaconda/Miniconda)



## Install Instructions

* Install Ubuntu Dependancies
```
sudo apt install git build-essential gcc-9 g++-9
```
For the viewer, also install:
```
sudo apt install xorg-dev
```
(There exists a headless mode without window management meant for training on a cluster, see below)

* Clone Repo
```
git clone git@github.com:lfranke/VET.git
cd VET/
git submodule update --init --recursive --jobs 0
```

* Create Conda Environment

```shell
cd VET
./create_environment.sh
```

* Install Pytorch

 ```shell
cd VET
./install_pytorch_precompiled.sh
```

* Compile VET

```shell
cd VET

conda activate vet

export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9

mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.9/site-packages/torch/;${CONDA}" ..

make -j10

```



## Running on pretrained models

Supplemental materials link: [https://zenodo.org/records/10477744](https://zenodo.org/records/10477744)

After a successful compilation, the best way to get started is to run `viewer` on the *tanks and temples* scenes
using our pretrained models.
First, download the scenes and extract them
into `scenes/`.
Now, download the model checkpoints and extract
them into `experiments/`.
Your folder structure should look like this:

```shell
VET/
    build/
        ...
    scenes/
        tt_train/
        tt_playground/
        ...
    experiments/
        checkpoint_train_vet
        checkpoint_playground_vet
        ...
```

## Viewer

Start the viewer with

```shell
conda activate vet
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA/lib
./build/bin/viewer --scene_dir scenes/tt_train

```
Your working directory should be the vet root directory.


## Scene Description

* VET uses ADOP's scene format.
* ADOP uses a simple, text-based scene description format.
* To run on your scenes you have to convert them into this format.
* If you have created your scene with COLMAP (like us) you can use the colmap2adop converter.
* More infos on this topic can be found here: [scenes/README.md](scenes/README.md)

## Training

The pipeline is fitted to your scenes by the `train` executable.
All training parameters are stored in a separate config file.
The basic syntax is:

```shell
./build/bin/adop_train --config configs/config.ini
```

Make again sure that the working directory is the root.
Otherwise, the loss models will not be found.

## Headless Mode

If you do not want the viewer application, consider calling cmake with an additional `-DHEADLESS`.


## Troubleshooting

* VET is build upon [ADOP](https://github.com/darglein/ADOP), take a look at the instructions there as well.

## License

The code here is licensed under MIT, however note that some submodules are not and are compiled against.
