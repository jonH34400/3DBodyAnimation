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