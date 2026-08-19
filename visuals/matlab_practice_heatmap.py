import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Months (columns)
months = [
    "Aug\n2025", "Sep", "Oct", "Nov", "Dec",
    "Jan", "Feb", "Mar", "Apr", "May",
    "Jun", "Jul", "Aug\n2026",
]

# Skills (rows)
skills = ["Basics", "Plotting", "Loops", "Vectors &\nMatrices", "Functions", "Debugging"]

# Intensity matrix (0-10): rows = skills, cols = months
# Tells a learning story: start with Basics/Vectors -> Plotting/Loops -> Functions/Debugging
data = np.array([
    # Aug25 Sep Oct Nov Dec Jan Feb Mar Apr May Jun Jul Aug26
    [   8,    3,  2,  0,  0,  0,  0,  0,  0,  0,  2,  3,  2],  # Basics
    [   3,    0,  2,  0,  0,  0,  0,  0,  0,  0,  5,  8,  3],  # Plotting
    [   2,    0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  7,  4],  # Loops
    [   6,    2,  0,  0,  0,  0,  0,  0,  0,  0,  1,  6,  2],  # Vectors & Matrices
    [   1,    0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  5,  6],  # Functions
    [   0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  5],  # Debugging
])

# Light futuristic colormap: white -> soft cyan -> deeper cyan
cmap = LinearSegmentedColormap.from_list(
    "cyan_soft", ["#FFFFFF", "#D6F0F5", "#7FD4E3", "#3BA7C0"]
)

fig, ax = plt.subplots(figsize=(10, 4.5))

im = ax.imshow(data, cmap=cmap, vmin=0, vmax=10, aspect="auto")

# Thin separator lines between cells for a clean grid feel
ax.set_xticks(np.arange(len(months)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(skills)) - 0.5, minor=True)
ax.grid(which="minor", color="#FFFFFF", linewidth=1.5)
ax.tick_params(which="minor", length=0)

# Labels
ax.set_xticks(range(len(months)))
ax.set_xticklabels(months, fontsize=8, color="#5A6A75")
ax.set_yticks(range(len(skills)))
ax.set_yticklabels(skills, fontsize=9, color="#3A4A55")
ax.tick_params(axis="both", length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

# Title — caps, lavender to match the radar
ax.set_title("MATLAB PRACTICE INTENSITY", fontsize=13, color="#6E6CA8", pad=12)

# Minimal colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=8, colors="#5A6A75", length=0)
cbar.set_label("intensity", fontsize=8, color="#888888")

plt.tight_layout()
plt.savefig("matlab_practice_heatmap.png", dpi=200)