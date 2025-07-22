# Body Animation: SMPL Model Fitting from Video

This project implements a C++ pipeline that takes an ordinary video of a person and outputs a new video with a visualized SMPL mesh overlaid per frame.

The key component is an optimization loop using **Ceres Solver** to fit SMPL model parameters (pose, shape, and translation) to 2D keypoints detected in each frame.

## Main Steps
1. **Preprocessing (Python)**
   - Extract frames from input video.
   - Detect 2D keypoints using Mediapipe / OpenPose.

2. **Optimization (C++)**
   - For each frame:
     - Load 2D keypoints
     - Fit SMPL model using Ceres Solver
     - Save optimized parameters

3. **Postprocessing (C++)**
   - Render SMPL mesh overlay using OpenCV/Open3D
   - Reconstruct final video sequence

## Requirements

1. **C++ Dependencies**
    - Ceres Solver (1.14)
    - OpenCV
    - Open3D (optional, for better mesh visualization)
    - C++17 or newer

2. **Python Dependencies (for preprocessing)**
    Install with:
    ```bash
    pip install -r requirements.txt


---

## Setup (Ubuntu)

### 1 Core build toolchain
```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y \
    build-essential cmake gdb pkg-config

```


### 2 Third-party libraries 
```bash
sudo apt install -y \
    libeigen3-dev            \  
    libceres-dev             \ 
    libsuitesparse-dev       \  
    nlohmann-json3-dev 
```



### 3  Install **libtorch** (C++ PyTorch)
```bash
mkdir -p ~/opt && cd ~/opt
curl -L -o libtorch.zip \
  "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-without-deps-2.1.2%2Bcpu.zip"
unzip libtorch.zip && rm libtorch.zip

# make CMake find it automatically
echo 'export Torch_DIR=$HOME/opt/libtorch' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$Torch_DIR' >> ~/.bashrc
source ~/.bashrc   # reload variables in this shell
```

### 4  Clone project & pull sub-modules
The repo already lists external dependencies in **.gitmodules** (`external/SMPLpp`, `external/xtensor`). One command fetches everything:

```bash
git clone --recursive https://github.com/jonH34400/3DBodyAnimation.git
cd 3DBodyAnimation
git checkout NewModel3
git submodule update --init --recursive
```



## Model assets — SMPL models

### 1  Pull original `.npz` via **Git LFS**
```bash
git lfs pull
```

That downloads:

```
assets/raw/
│
├── basicModel_f_lbs_10_207_0_v1.0.0.npz
└── basicModel_m_lbs_10_207_0_v1.0.0.npz
```

### 2 (Or) Download from official website
1. Create a free account at **<http://smpl.is.tue.mpg.de>**  
2. Download one or more raw models:  
   * `basicModel_f_lbs_10_207_0_v1.0.0.npz`  – female  
   * `basicModel_m_lbs_10_207_0_v1.0.0.npz`  – male  


### 3 Preprocess `.npz` to `.npz`
   Run scripts/npz_fixer.py


## Preprocess model to `.json` (old)
   ```bash
   python scripts/convert_smplpp.py \
     assets/raw/basicModel_f_lbs_10_207_0_v1.0.0.npz \
     -o assets/models/SMPL_FEMALE.json
   ```
---

## Build & run
```bash
mkdir build && cd build
cmake .. -DCeres_DIR="../external/install/ceres-1.14/lib/cmake/Ceres" -DCMAKE_BUILD_TYPE=Release -DWITH_OMP=ON
make -j8
./3dba ../data/avatar-model/ ../data/keypoints/video2/frame_0060.json ../data/frames_annotated/video2/frame_0060_annotated.png
```
A successful build drops a `neutral_mesh.obj` T-pose.
