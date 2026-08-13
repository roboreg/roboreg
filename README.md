# roboreg
[![Unit tests](https://github.com/lbr-stack/roboreg/actions/workflows/tests.yaml/badge.svg)](https://github.com/lbr-stack/roboreg/actions/workflows/tests.yaml)
[![PyPI version](https://img.shields.io/pypi/v/roboreg.svg)](https://pypi.org/project/roboreg/)
[![License: Apache License 2.0](https://img.shields.io/github/license/lbr-stack/roboreg)](https://github.com/lbr-stack/roboreg?tab=Apache-2.0-1-ov-file)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Eye-to-hand calibration from RGB-D images using robot mesh as calibration target.

<body>
    <table>
    <caption>Mesh (purple) and Point Cloud (turqoise).</caption>
        <tr>
            <th align="left" width="50%">Unregistered</th>
            <th align="left" width="50%">Registered</th>
        </tr>
        <tr>
            <td align="center"><img src="https://raw.githubusercontent.com/lbr-stack/roboreg/refs/heads/main/doc/img/hydra_robust_icp_unregistered.png" alt="Unregistered Mesh and Point Cloud"></td>
            <td align="center"><img src="https://raw.githubusercontent.com/lbr-stack/roboreg/refs/heads/main/doc/img/hydra_robust_icp_registered.png" alt="Registered Mesh and Point Cloud"></td>
        </tr>
    </table>
</body>

## Table of Contents
- [Installation](#installation)
    - [Pip (Requires CUDA Toolkit Installation)](#pip-requires-cuda-toolkit-installation)
    - [Docker (Comes with CUDA Toolkit)](#docker-comes-with-cuda-toolkit)
- [Command Line Interface](#command-line-interface)
    - [Segment](#segment)
    - [Hydra Robust ICP](#hydra-robust-icp)
    - [Camera Swarm](#camera-swarm)
    - [Monocular Differentiable Rendering](#monocular-differentiable-rendering)
    - [Stereo Differentiable Rendering](#stereo-differentiable-rendering)
    - [Render Results](#render-results)
- [Testing](#testing)

## Installation
Two install options are provided: 
- [Pip (Requires CUDA Toolkit Installation)](#pip-requires-cuda-toolkit-installation)
- [Docker (Comes with CUDA Toolkit)](#docker-comes-with-cuda-toolkit)

### Pip (Requires CUDA Toolkit Installation)
> [!NOTE]
> During runtime, CUDA Toolkit is required for the differentiable rendering. If you are planning to do differentiable rendering, see [CUDA Toolkit Install Instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

To `pip` intall `roboreg`, simply run

```shell
pip install roboreg
```

### Docker (Comes with CUDA Toolkit)
A sample Docker container is provided for testing purposes. First:

- Install Docker, see [Docker Install Instructions](https://docs.docker.com/engine/install/)
- Install NVIDIA Container Toolkit, see [NVIDIA Container Toolkit Install Instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Next: 

1. Clone this repository

    ```shell
    git clone git@github.com:lbr-stack/roboreg.git
    ```

2. Build the Docker image (currently only runtime support, i.e. no rendering via compiled kernels)

    ```shell
    cd roboreg
    docker build \
        -t roboreg:latest \
        -f .docker/Dockerfile \
        --build-arg PYTORCH_VERSION=2.10.0 \
        --build-arg CUDA_VERSION=13.0 \
        .
    ```

    For CUDA 12.4, use:

    ```shell
    docker build \
        -t roboreg:latest \
        -f .docker/Dockerfile \
        --build-arg PYTORCH_VERSION=2.6.0 \
        --build-arg CUDA_VERSION=12.4 \
        .
    ```

3. Run container (on Linux host with [NVIDIA Container Toolkit](#docker-comes-with-cuda-toolkit))

    ```shell
    docker run -it \
        --gpus all \
        --network host \
        --ipc host \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume /dev/shm:/dev/shm \
        --volume /dev:/dev --privileged \
        --env DISPLAY \
        --env QT_X11_NO_MITSHM=1 \
        roboreg:latest
    ```

## Command Line Interface
> [!TIP]
> Examples use sample data under [test/assets/lbr_med7_r800](test/assets/lbr_med7_r800). Data is stored via Git Large File Storage (LFS). Data is cloned automatically when `git-lfs` is installed. To clone in retrospect:
> ```shell
> sudo apt install git-lfs
> git lfs fetch --all
> git lfs checkout
> ```

### Segment
This is a required step to generate robot masks.

```shell
rr-sam2 \
    --path test/assets/lbr_med7_r800/samples \
    --pattern "left_image_*.png" \
    --n-positive-annotations 5 \
    --n-negative-annotations 5 \
    --device cuda
```

### Hydra Robust ICP
The Hydra robust ICP implements a point-to-plane ICP registration on a Lie algebra. It does not use rendering and can also be used on CPU.

```shell
rr-hydra \
    --intrinsics-file test/assets/lbr_med7_r800/samples/left_camera_info.yaml \
    --path test/assets/lbr_med7_r800/samples \
    --mask-pattern mask_sam2_left_image_*.png \
    --depth-pattern depth_*.npy \
    --joint-states-pattern joint_states_*.npy \
    --urdf-path test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --number-of-points 5000 \
    --display-results \
    --output-file HT_hydra_robust.csv
```

### Camera Swarm
> [!WARNING]
> On first run, `nvdiffrast` compiles PyTorch extensions. This might use too many resources on some systems (< 16 GB RAM). 
> You can create an environment variable `export MAX_JOBS=1` before the first run to limit concurrent compilation.
> Also refer to this [Issue](https://github.com/NVlabs/nvdiffrast/issues/201).

The camera swarm optimization can serve for finding an initial guess to [Monocular Differentiable Rendering](#monocular-differentiable-rendering) or [Stereo Differentiable Rendering](#stereo-differentiable-rendering).

```shell
rr-cam-swarm \
    --collision-meshes \
    --n-cameras 1000 \
    --min-distance 0.5 \
    --max-distance 3.0 \
    --angle-range 3.141 \
    --w 0.7 \
    --c1 1.5 \
    --c2 1.5 \
    --max-iterations 100 \
    --display-progress \
    --urdf-path test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --target-reduction 0.8 \
    --scale 0.1 \
    --n-samples 1 \
    --camera-info-file test/assets/lbr_med7_r800/samples/left_camera_info.yaml \
    --path test/assets/lbr_med7_r800/samples \
    --image-pattern left_image_*.png \
    --joint-states-pattern joint_states_*.npy \
    --mask-pattern mask_sam2_left_image_*.png \
    --output-file HT_cam_swarm.csv
```

### Monocular Differentiable Rendering
> [!WARNING]
> On first run, `nvdiffrast` compiles PyTorch extensions. This might use too many resources on some systems (< 16 GB RAM). 
> You can create an environment variable `export MAX_JOBS=1` before the first run to limit concurrent compilation.
> Also refer to this [Issue](https://github.com/NVlabs/nvdiffrast/issues/201).

This monocular differentiable rendering refinement requires a good initial estimate, as e.g. obtained from [Hydra Robust ICP](#hydra-robust-icp) or [Camera Swarm](#camera-swarm)

```shell
rr-mono-dr \
    --optimizer AdamW \
    --lr 0.03 \
    --max-iterations 400 \
    --convergence-tolerance 0.001 \
    --convergence-patience 100 \
    --scheduler-factor 0.1 \
    --scheduler-patience 40 \
    --scheduler-threshold 0.0001 \
    --display-progress \
    --urdf-path test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --intrinsics-file test/assets/lbr_med7_r800/samples/left_camera_info.yaml \
    --extrinsics-file test/assets/lbr_med7_r800/samples/HT_hydra_robust.csv \
    --path test/assets/lbr_med7_r800/samples \
    --image-pattern left_image_*.png \
    --joint-states-pattern joint_states_*.npy \
    --mask-pattern mask_sam2_left_image_*.png \
    --output-file HT_dr.csv
```

### Stereo Differentiable Rendering
> [!WARNING]
> On first run, `nvdiffrast` compiles PyTorch extensions. This might use too many resources on some systems (< 16 GB RAM). 
> You can create an environment variable `export MAX_JOBS=1` before the first run to limit concurrent compilation.
> Also refer to this [Issue](https://github.com/NVlabs/nvdiffrast/issues/201).

This stereo differentiable rendering refinement requires a good initial estimate, as e.g. obtained from [Hydra Robust ICP](#hydra-robust-icp) or [Camera Swarm](#camera-swarm)

```shell
rr-stereo-dr \
    --optimizer AdamW \
    --lr 0.03 \
    --max-iterations 400 \
    --convergence-tolerance 0.001 \
    --convergence-patience 100 \
    --scheduler-factor 0.1 \
    --scheduler-patience 40 \
    --scheduler-threshold 0.0001 \
    --display-progress \
    --urdf-path test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --left-intrinsics-file test/assets/lbr_med7_r800/samples/left_camera_info.yaml \
    --right-intrinsics-file test/assets/lbr_med7_r800/samples/right_camera_info.yaml \
    --left-extrinsics-file test/assets/lbr_med7_r800/samples/HT_hydra_robust.csv \
    --right-extrinsics-file test/assets/lbr_med7_r800/samples/HT_right_to_left.csv \
    --path test/assets/lbr_med7_r800/samples \
    --left-image-pattern left_image_*.png \
    --right-image-pattern right_image_*.png \
    --joint-states-pattern joint_states_*.npy \
    --left-mask-pattern mask_sam2_left_image_*.png \
    --right-mask-pattern mask_sam2_right_image_*.png \
    --left-output-file HT_left_dr.csv \
    --right-output-file HT_right_dr.csv
```

### Render Results
> [!WARNING]
> On first run, `nvdiffrast` compiles PyTorch extensions. This might use too many resources on some systems (< 16 GB RAM). 
> You can create an environment variable `export MAX_JOBS=1` before the first run to limit concurrent compilation.
> Also refer to this [Issue](https://github.com/NVlabs/nvdiffrast/issues/201).

Generate renders using the obtained extrinsics:

```shell
rr-render \
    --batch-size 1 \
    --num-workers 0 \
    --urdf-path test/assets/lbr_med7_r800/description/lbr_med7_r800.urdf \
    --root-link-name lbr_link_0 \
    --end-link-name lbr_link_7 \
    --camera-info-file test/assets/lbr_med7_r800/samples/left_camera_info.yaml \
    --extrinsics-file test/assets/lbr_med7_r800/samples/HT_left_dr.csv \
    --images-path test/assets/lbr_med7_r800/samples \
    --joint-states-path test/assets/lbr_med7_r800/samples \
    --image-pattern left_image_*.png \
    --joint-states-pattern joint_states_*.npy \
    --output-path /tmp/renders/lbr_med7_r800
```

## Testing
For testing on the `xarm` data, follow [Docker (Comes with CUDA Toolkit)](#docker-comes-with-cuda-toolkit). Inside the container, do

### Hydra Robust ICP
To run Hydra robust ICP on provided `xarm` and `realsense` data, run

```shell
rr-hydra \
    --intrinsics-file test/assets/xarm_7/samples/camera_info.yaml \
    --path test/assets/xarm_7/samples \
    --mask-pattern mask_*.png \
    --depth-pattern depth_*.npy \
    --joint-states-pattern joint_state_*.npy \
    --urdf-path test/assets/xarm_7/description/xarm_7.urdf \
    --root-link-name link_base \
    --end-link-name link7 \
    --number-of-points 5000 \
    --display-results \
    --output-file HT_hydra_robust.csv
```

### Render Results
Generate renders using the obtained extrinsics:

```shell
rr-render \
    --batch-size 1 \
    --num-workers 0 \
    --urdf-path test/assets/xarm_7/description/xarm_7.urdf \
    --root-link-name link_base \
    --end-link-name link7 \
    --camera-info-file test/assets/xarm_7/samples/camera_info.yaml \
    --extrinsics-file test/assets/xarm_7/samples/HT_hydra_robust.csv \
    --images-path test/assets/xarm_7/samples \
    --joint-states-path test/assets/xarm_7/samples \
    --image-pattern img_*.png \
    --joint-states-pattern joint_state_*.npy \
    --output-path /tmp/renders/xarm_7
```

## Acknowledgements
### Organizations and Grants
We would further like to acknowledge following supporters:

| Logo | Notes |
|:--:|:---|
| <img src="https://medicalengineering.org.uk/wp-content/themes/aalto-child/_assets/images/medicalengineering-logo.svg" alt="wellcome" width="150" align="left">  | This work was supported by core and project funding from the Wellcome/EPSRC [WT203148/Z/16/Z; NS/A000049/1; WT101957; NS/A000027/1]. |
| <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/1920px-Flag_of_Europe.svg.png" alt="eu_flag" width="150" align="left"> | This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101016985 (FAROS project). |
| <img src="https://rvim.online/author/avatar_hu8970a6942005977dc117387facf47a75_62303_270x270_fill_lanczos_center_2.png" alt="RViMLab" width="150" align="left"> | Built at [RViMLab](https://rvim.online/). |
| <img src="https://avatars.githubusercontent.com/u/75276868?s=200&v=4" alt="King's College London" width="150" align="left"> | Built at [CAI4CAI](https://cai4cai.ml/). |
| <img src="https://upload.wikimedia.org/wikipedia/commons/1/14/King%27s_College_London_logo.svg" alt="King's College London" width="150" align="left"> | Built at [King's College London](https://www.kcl.ac.uk/). |
