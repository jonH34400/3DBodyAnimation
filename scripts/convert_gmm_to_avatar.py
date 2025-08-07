#!/usr/bin/env python3
# convert_gmm08_to_pose_prior_txt.py  gmm_08.pkl  pose_prior.txt
import sys, pickle, numpy as np
if len(sys.argv) < 3:
    print("Usage:  convert_gmm08_to_pose_prior_txt.py  gmm_08.pkl  pose_prior.txt")
    sys.exit(1)

src, dst = sys.argv[1:3]
gmm = pickle.load(open(src, 'rb'), encoding='latin1')
means, covs, weights = gmm['means'], gmm['covars'], gmm['weights']
K, D = means.shape
assert D == 69, f"Expected 69-D pose, got {D}"

with open(dst, "w") as f:
    # 1. header ------------------------------------------------------------
    f.write(f"{K} {D}\n")

    # 2. weights -----------------------------------------------------------
    f.write(" ".join(map(str, weights)) + "\n")

    # 3. means -------------------------------------------------------------
    for k in range(K):
        f.write(" ".join(map(str, means[k])) + "\n")

    # 4. full covariance matrices -----------------------------------------
    #    row-major, one matrix after another
    for k in range(K):
        C = covs[k].reshape(D, D)
        f.write(" ".join(map(str, C.flatten())) + "\n")

print("✓ pose_prior.txt written:",
      K, "components,", D, "dims each")
