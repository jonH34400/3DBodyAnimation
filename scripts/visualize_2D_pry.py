import json
import matplotlib.pyplot as plt

# Load data
with open('joints_projection.json') as f:
    data = json.load(f)

joints = data['joints']

# Extract X and Y coordinates
x_coords = [j['x'] for j in joints]
y_coords = [j['y'] for j in joints]

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_coords, y_coords, c='red', zorder=2)

# Invert Y axis to match image coordinate system (y-down)
ax.invert_yaxis()

# Annotate joint indices
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    ax.text(x, y, str(i), fontsize=8, ha='center', va='bottom', zorder=3)

# Set aspect ratio to 1:1
ax.set_aspect('equal', adjustable='datalim')  # ✅ Important fix

# Titles and labels
ax.set_title("2D Joint Projections")
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.grid(True)

# Save to file
plt.tight_layout()
plt.savefig('joints_2d.png')
print("Visualization saved to 'joints_2d.png'")