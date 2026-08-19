import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

months = [
    "Aug\n2025", "Sep", "Oct", "Nov", "Dec",
    "Jan", "Feb", "Mar", "Apr", "May",
    "Jun", "Jul", "Aug\n2026"
]
days_active = [4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 8, 4]

# Light futuristic palette — soft cyan / lavender
BAR_FACE = "#7FD4E3"   # soft cyan
BAR_EDGE = "#3BA7C0"   # deeper cyan outline (wireframe look)

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    months, days_active, width=0.22,
    color=BAR_FACE, edgecolor=BAR_EDGE, linewidth=1.2,
    alpha=0.45, zorder=3
)

ax.set_title("MATLAB ONRAMP PROGRESS HISTORY", fontsize=13)
ax.set_ylim(0, 10)
ax.tick_params(axis="x", labelsize=8, length=0, colors="#5A6A75")
ax.tick_params(axis="y", labelsize=9, length=0, colors="#5A6A75")

# Full grid background — both axes
ax.grid(True, color="#E8ECF0", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#C8D0D8")

plt.tight_layout()
plt.savefig("matlab_onramp_progress.png", dpi=200)