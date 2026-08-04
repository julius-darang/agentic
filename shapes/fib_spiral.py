"""Create a 2D Fibonacci spiral, also called a sunflower seed pattern.

Run with:
    python fib_spiral.py

Each point is given a radius and an angle. The angle advances by the golden
angle each time, which spreads neighboring points around the disk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
POINT_COUNT = 5_000


def main() -> None:
    # Start just away from the center and expand toward the edge of the disk.
    index = np.arange(POINT_COUNT) + 0.5
    radius = np.sqrt(index / POINT_COUNT)
    angle = index * GOLDEN_ANGLE

    # Convert polar coordinates to Cartesian coordinates.
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=3, color="tab:blue", alpha=0.75)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.title("Fibonacci spiral using the golden angle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.2)
    output_path = OUTPUT_DIR / "fib_spiral.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
