"""Place Fibonacci-spiral points on the surface of a sphere.

Run with:
    python fib_spiral_sphere.py

This version only draws the flat (x, y) projection. The z coordinate is
calculated but intentionally ignored, so there is no depth shading yet.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
POINT_COUNT = 5_000


def fibonacci_surface(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evenly distributed points on a unit sphere."""
    i = np.arange(n, dtype=float) + 0.5
    y = 1.0 - 2.0 * i / n
    ring = np.sqrt(1.0 - y * y)
    theta = GOLDEN_ANGLE * i

    x = ring * np.cos(theta)
    z = ring * np.sin(theta)
    return x, y, z


def main() -> None:
    x, y, z = fibonacci_surface(POINT_COUNT)

    # Flat scatter: z is calculated for the sphere, but ignored for now.
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=3, color="tab:blue", alpha=0.75)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.title("Fibonacci points on a sphere, flat projection")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.2)
    output_path = OUTPUT_DIR / "fib_spiral_sphere.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
