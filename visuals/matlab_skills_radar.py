import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Skill areas + self-rating (0-10). Edit these freely.
skills = [
    "Basics", "Plotting", "Loops",
    "Vectors &\nMatrices", "Functions", "Debugging",
]
ratings = [8, 7, 7, 9, 6, 5]

# Light futuristic palette to match the Onramp chart
FILL = "#7FD4E3"
EDGE = "#3BA7C0"
GRID = "#E8ECF0"
SPOKE = "#D6DEE6"

N = len(skills)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# Closed loops
values = ratings + [ratings[0]]
angles_closed = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Polygon — low opacity fill + crisp outline (wireframe feel)
ax.plot(angles_closed, values, color=EDGE, linewidth=1.4, zorder=3)
ax.fill(angles_closed, values, color=FILL, alpha=0.35, zorder=3)

# Axis setup
ax.set_theta_offset(np.pi / 2)   # start at top
ax.set_theta_direction(-1)        # clockwise
ax.set_rlim(0, 10)
ax.set_rticks([2, 4, 6, 8, 10])
ax.tick_params(axis="y", labelsize=8, colors="#5A6A75")
ax.set_rlabel_position(90)        # radial labels on the vertical axis

# --- Make the chart a hexagon, not a circle ---
ax.grid(False)                              # turn off circular gridlines
ax.spines["polar"].set_visible(False)       # hide the round outer spine

# Concentric hexagon gridlines at each rtick
rings = [2, 4, 6, 8, 10]
for r in rings:
    xs = [a for a in angles_closed]
    ys = [r] * (N + 1)
    ax.plot(xs, ys, color=GRID, linewidth=0.8, zorder=1)

# Spokes from center to each outer vertex
for a in angles:
    ax.plot([a, a], [0, 10], color=SPOKE, linewidth=0.8, zorder=1)

# Skill labels around the perimeter
ax.set_xticks(angles)
ax.set_xticklabels(skills, fontsize=9, color="#3A4A55")

# Title — caps to match the other chart
ax.set_title("MATLAB SKILLS SNAPSHOT", fontsize=13, color="#6E6CA8", pad=22)

plt.tight_layout()
plt.savefig("matlab_skills_radar.png", dpi=200)