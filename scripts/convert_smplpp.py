
#!/usr/bin/env python3
import argparse, json, pathlib, numpy as np

def ensure(arr, shape, name):
    assert arr.shape == shape, f"{name}: expected {shape}, got {arr.shape}"
    return arr


def main():
    p = argparse.ArgumentParser(description="Convert SMPL NPZ to SMPL++ JSON")
    p.add_argument("npz")
    p.add_argument("-o", "--out", default="SMPL_MODEL.json")
    args = p.parse_args()

    d = np.load(args.npz)

    vt = ensure(d["v_template"], (6890, 3), "v_template").astype(np.float32)  # (6890,3)

    sd_raw = d["shapedirs"]
    sd = sd_raw.astype(np.float32).reshape(6890, 3, 10)
   
    # (6890,3,10)

    pd_raw = d["posedirs"]
    pd = pd_raw.astype(np.float32).reshape(6890, 3, 207)
  
    # (6890,3,207)

    jr = d["J_regressor"]
    if jr.shape == (6890, 24):
        jr = jr.T
    jr = ensure(jr, (24, 6890), "J_regressor").astype(np.float32)               # (24,6890)

    wt = ensure(d["weights"], (6890, 24), "weights").astype(np.float32)       # (6890,24)
    fi = ensure(d["f"], (13776, 3), "f").astype(np.int32)
    kt = ensure(d["kintree_table"], (2, 24), "kintree_table").astype(np.int64) # (2,24)

    out = {
        "vertices_template": vt.tolist(),
        "shape_blend_shapes": sd.tolist(),
        "pose_blend_shapes": pd.tolist(),
        "joint_regressor": jr.tolist(),
        "weights": wt.tolist(),
        "face_indices": fi.tolist(),
        "kinematic_tree": kt.tolist(),
    }

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print("wrote", dest.resolve())


if __name__ == "__main__":
    main()


