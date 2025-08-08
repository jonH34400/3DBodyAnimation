import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python eval.py <csv_path>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

mean_pixel_error = df['mean_pixel_error_px'].mean()
# Convert ms to s
opt_time_s = df['time_ms'].mean() / 1e3

print(f"Mean Pixel Error: {mean_pixel_error:.4f}")
print(f"Optimization Time (s): {opt_time_s:.4f}")
