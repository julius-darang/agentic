"""Compare two ways to generate random points inside a disk.

Run with:
    python uniform_points.py

The first method uses a random radius directly. It creates a dense center
because small-radius regions are represented too often. The second method
uses sqrt(random()) so the points are uniform by area.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
POINT_COUNT = 5_000
SEED = 7


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Use the same random values for both methods so the difference comes only
    # from how the radius is calculated.
    random_radius = rng.random(POINT_COUNT)
    theta = rng.uniform(0.0, 2.0 * np.pi, POINT_COUNT)

    # Naive method: this bunches points toward the center.
    naive_radius = random_radius
    naive_x = naive_radius * np.cos(theta)
    naive_y = naive_radius * np.sin(theta)

    # Correct method: sqrt() accounts for the fact that outer rings have more
    # area than inner rings, producing a uniform density across the disk.
    uniform_radius = np.sqrt(random_radius)
    uniform_x = uniform_radius * np.cos(theta)
    uniform_y = uniform_radius * np.sin(theta)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(naive_x, naive_y, s=3, color="tab:blue", alpha=0.45)
    axes[0].set_title("Naive: r = random()")

    axes[1].scatter(uniform_x, uniform_y, s=3, color="tab:blue", alpha=0.45)
    axes[1].set_title("Correct: r = sqrt(random())")

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.2)

    figure.suptitle("Random points inside a disk")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "uniform_points.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
