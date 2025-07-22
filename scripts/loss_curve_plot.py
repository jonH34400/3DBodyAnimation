import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loss_curve.txt")
plt.plot(df["iteration"], df["loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve - Optimization of SMPL")
plt.grid(True)
# Save to file
plt.tight_layout()
plt.savefig('loss_curve.png')
print("Visualization saved to 'loss_curve.png'")