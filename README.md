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
    - Ceres Solver
    - OpenCV
    - Open3D (optional, for better mesh visualization)
    - C++17 or newer

2. **Python Dependencies (for preprocessing)**
    Install with:
    ```bash
    pip install -r requirements.txt


## Local Setup (WSL 2)

### 1 Core build toolchain
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake gdb pkg-config autotools-dev
```

---

### 2 Third-party libraries (apt)
```bash
sudo apt install -y \
    libgflags-dev libgoogle-glog-dev \
    libatlas-base-dev libeigen3-dev libsuitesparse-dev \
    libceres-dev \
    libicu-dev libbz2-dev libboost-all-dev \
    libflann-dev libfreeimage-dev liblz4-dev \
    nlohmann-json3-dev  
```

---

### 3  Install **libtorch** (C++ PyTorch)
```bash
mkdir -p ~/opt && cd ~/opt
curl -L -o libtorch.zip \
  "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.2%2Bcu121.zip"
unzip libtorch.zip && rm libtorch.zip

# make CMake find it automatically
echo 'export Torch_DIR=$HOME/opt/libtorch' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$Torch_DIR' >> ~/.bashrc
source ~/.bashrc   # reload variables in this shell
```

> **Why libtorch and not Conda?**  
> SMPLpp is pure-C++; the libtorch zip ships the C++ headers & shared libs that CMake can detect. Conda installs the Python wheel—hard to link from C++.

---

### 4  Clone project & pull sub-modules
The repo already lists external dependencies in **.gitmodules** (`external/SMPLpp`, `external/xtensor`). One command fetches everything:

```bash
git clone --recursive git@github.com:jonH34400/3DBodyAnimation.git
cd 3DBodyAnimation
```

---

### 5  Build & run
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=$Torch_DIR ..
make -j$(nproc)
./3dsmc  
```
A successful build drops a `test.obj` T-pose.

