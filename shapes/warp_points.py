"""Compare pole concentration by warping a Fibonacci sphere distribution.

Run with:
    python warp_points.py

The latitude is changed with:
    y = sign(y) * abs(y) ** p

Values below 1 move points toward the poles; values above 1 move them toward
the equator. The three values are rendered side by side.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
POINT_COUNT = 5_000
P_VALUES = (0.25,)


def warped_surface(
    n: int, p: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Fibonacci sphere whose latitude distribution is warped."""
    i = np.arange(n, dtype=float) + 0.5
    base_y = 1.0 - 2.0 * i / n

    # Warp latitude before calculating the sphere's x/z ring.
    y = np.sign(base_y) * np.abs(base_y) ** p
    ring = np.sqrt(1.0 - y * y)
    theta = GOLDEN_ANGLE * i
    x = ring * np.cos(theta)
    z = ring * np.sin(theta)
    return x, y, z


def main() -> None:
    figure, axes = plt.subplots(
        1, len(P_VALUES), figsize=(7 * len(P_VALUES), 7), facecolor="white"
    )
    axes = np.atleast_1d(axes)

    for axis, p in zip(axes, P_VALUES):
        x, y, z = warped_surface(POINT_COUNT, p)

        # Reuse the depth mapping: front points are brighter and larger.
        front = (z + 1.0) * 0.5
        axis.scatter(
            x,
            y,
            s=2.0 + 18.0 * front,
            color="tab:blue",
            alpha=0.85,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_title(f"p = {p}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.2)

    figure.suptitle("Warped Fibonacci sphere distribution")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "warp_points.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
