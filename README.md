# Body Animation: SMPL Model Fitting from Video
The code is available [here](https://github.com/jonH34400/3DBodyAnimation).

This [project](https://github.com/jonH34400/3DBodyAnimation) implements a C++ pipeline that takes an ordinary video of a person and outputs a new video with a visualized SMPL mesh overlaid per frame.

The key component is an optimization loop using **Ceres Solver** to fit SMPL model parameters (pose, shape, and transformation) to 2D keypoints detected in each frame.

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



### 3 Clone project & pull [Avatar](https://github.com/sxyu/avatar/tree/master) sub-module
```bash
git clone --recursive https://github.com/jonH34400/3DBodyAnimation.git
cd 3DBodyAnimation
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
   Run `scripts/npz_fixer.py` and placce `output model.npz` into `data/avatar-model/`.
### 4 Video Data and Keypoint Detection
   Download any video from YouTube and detect 2d keypoints with `data/extract_key_kpoints_mediapipe.py`
---

## Build & run
```bash
mkdir build && cd build
cmake .. -DCeres_DIR="../external/install/ceres-1.14/lib/cmake/Ceres" -DCMAKE_BUILD_TYPE=Release -DWITH_OMP=ON
make -j
./3dba_single \
    ../data/avatar-model/              # [1] Path to SMPL.npz model file
    ../data/keypoints/dancing_dude/    # [2] Folder with keypoint JSON files
    ../data/frames_annotated/dancing_dude/  # [3] Folder with annotated input images
    ../data/out/dancing_dude_sf_shape_gmm   # [4] Output folder for logs & renders
    --use-gmm                           # [flag] Enable GMM pose prior
    --opt-shape                         # [flag] Optimize body shape as well as pose

./3dba_multi \
    ../data/avatar-model/                   # [1] Path to SMPL.npz model file
    ../data/keypoints/dancing_dude/         # [2] Folder with keypoint JSON files
    ../data/frames_annotated/dancing_dude/  # [3] Folder with annotated input images
    ../data/out/dancing_dude_mf_shape_gmm   # [4] Output folder for logs & renders
    1000                                    # [5] max_iters for stage-1 (anchor frames)
    500                                     # [6] max_iters for stage-2 (window refinement)
    10                                      # [7] anchor_skip → spacing between anchor frames
    20                                      # [8] window size for sliding-window optimization
    5                                       # [9] window overlap
    20.0                                    # [10] β_pose (pose prior weight)
    30.0                                    # [11] β_shape (shape prior weight)
    3.0                                     # [12] λ_temp (temporal smoothness weight)

```
