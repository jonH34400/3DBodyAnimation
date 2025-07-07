#!/usr/bin/env python3
import numpy as np
import sys, os
from os import path

def main(argv):
    if len(argv) < 3:
        print("Usage: smpl2npz.py <out_dir> <model1.npz> [...]", file=sys.stderr)
        sys.exit(1)

    out_dir = argv[1]
    os.makedirs(out_dir, exist_ok=True)

    for in_path in argv[2:]:
        print(f"\nLoading     : {in_path}")
        npz = np.load(in_path, allow_pickle=True)
        print("Original fields and shapes:")
        for key in npz.files:
            print(f"  {key}: {npz[key].shape}")

        data = {k: npz[k] for k in npz.files}

        # Only reshape posedirs (207 × 3*6890) → (6890,3,207)
        if 'posedirs' in data and data['posedirs'].ndim == 2:
            arr = data['posedirs']
            n_blends, flat = arr.shape
            n_verts = flat // 3
            new_shape = (n_verts, 3, n_blends)
            # transpose first so that each block of 3 coords is together
            data['posedirs'] = arr.T.reshape(new_shape)
            print(f"→ Reshaped 'posedirs' to {new_shape}")

        # everything else stays exactly as in the original NPZ,
        # including shapedirs which was already (6890,3,10)

        fname    = path.basename(in_path)
        out_path = path.join(out_dir, fname)
        print(f"Writing    : {out_path} (uncompressed)")
        # write uncompressed so cnpy can read it
        np.savez(out_path, **data)

    print("\nDone.\n")

if __name__ == '__main__':
    main(sys.argv)
