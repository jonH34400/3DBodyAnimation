import numpy as np

num_joints = 24
dim = num_joints * 3
num_components = 5

# Generate few basic poses
def generate_pose(name):
    pose = np.ones(dim)
    if name == "t_pose":
        # completely extended arms
        pose[16*3 + 1] = -1.0  # l-shoulder Y
        pose[17*3 + 1] = 1.0   # r-shoulder Y
        pose[18*3 + 1] = -1.0  # l-elbow Y
        pose[19*3 + 1] = 1.0   # r-elbow Y
    elif name == "crouch":
        pose[4*3 + 0] = 1.0   # l-knee X
        pose[5*3 + 0] = 1.0   # r-knee X
    elif name == "arms_down":
        pose[16*3 + 0] = -0.2
        pose[17*3 + 0] = 0.2
    elif name == "arms_forward":
        pose[16*3 + 2] = 0.8
        pose[17*3 + 2] = 0.8
        pose[18*3 + 2] = 0.5
        pose[19*3 + 2] = 0.5
    elif name == "slight_twist":
        pose[1*3 + 1] = 0.3   # l-hip Y
        pose[2*3 + 1] = -0.3  # r-hip Y
    return pose

# Defining sizes of each component
pose_names = ["t_pose", "crouch", "arms_down", "arms_forward", "slight_twist"]
means = [generate_pose(name) for name in pose_names]

# Small Covariance Matrix
covariance = np.eye(dim) * 0.01

# Saving file
with open("../data/avatar-model/pose_prior.txt", "w") as f:
    f.write(f"{num_components} {dim}\n")

    #uniform weights
    for _ in range(num_components):
        f.write("1.0\n")

    # means
    for mean in means:
        f.write(" ".join(map(str, mean)) + "\n")

    # covariances
    for _ in range(num_components):
        for i in range(dim):
            f.write(" ".join(map(str, covariance[i])) + "\n")
