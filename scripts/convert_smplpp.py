#!/usr/bin/env python3
"""
Convert an official SMPL .npz (or .pkl) to the layout SMPLpp expects.

Input keys  :  v_template, f, weights, shapedirs, posedirs,
               J_regressor, kintree_table
Output keys :  vertices_template, face_indices (1-based),
               weights, shape_blend_shapes, pose_blend_shapes,
               joint_regressor, kinematic_tree
"""

import argparse, numpy as np, pickle, os, pathlib, sys, scipy.sparse as sp

def load_model(path):
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
    elif path.suffix == ".pkl":
        # raw pickle can be py2 or py3; allow both
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    else:
        sys.exit(f"Unsupported extension: {path.suffix}")
    return data

def dense(arr):
    return arr.toarray() if isinstance(arr, sp.spmatrix) else arr

def convert(raw):
    return dict(
        vertices_template = np.asarray(raw["v_template"]),
        face_indices      = np.asarray(raw["f"]) + 1,        # 0->1 indexing
        weights           = np.asarray(raw["weights"]),
        shape_blend_shapes= np.asarray(raw["shapedirs"]),
        pose_blend_shapes = np.asarray(raw["posedirs"]),
        joint_regressor   = dense(raw["J_regressor"]),
        kinematic_tree    = np.asarray(raw["kintree_table"]),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="raw SMPL .pkl or .npz")
    ap.add_argument("--output", required=True, help="destination .npz")
    args = ap.parse_args()

    raw   = load_model(pathlib.Path(args.input))
    model = convert(raw)
    np.savez(args.output, **model)
    print("Wrote", os.path.abspath(args.output))

if __name__ == "__main__":
    main()
