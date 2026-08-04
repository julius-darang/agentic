"""Show sphere depth in a flat scatter plot.

Run with:
    python depth_points.py

The z coordinate is mapped to point color and size. No 3D renderer is used.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
POINT_COUNT = 5_000


def fibonacci_surface(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.arange(n, dtype=float) + 0.5
    y = 1.0 - 2.0 * i / n
    ring = np.sqrt(1.0 - y * y)
    theta = GOLDEN_ANGLE * i
    x = ring * np.cos(theta)
    z = ring * np.sin(theta)
    return x, y, z


def main() -> None:
    x, y, z = fibonacci_surface(POINT_COUNT)

    # Map z from [-1, 1] to [0, 1]: 0 is back, 1 is front.
    front = (z + 1.0) * 0.5
    point_size = 2.0 + 18.0 * front

    figure, axis = plt.subplots(figsize=(7, 7), facecolor="white")
    axis.scatter(
        x,
        y,
        s=point_size,
        color="tab:blue",
        alpha=0.85,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_title("Sphere depth shown by color and point size")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True, alpha=0.2)
    output_path = OUTPUT_DIR / "depth_points.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
