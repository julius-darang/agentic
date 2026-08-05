"""Add grain and a vignette to a rasterized Fibonacci sphere.

Run with:
    python grain_vignette.py

The Gaussian splats are painted first. Then random grain and a radial falloff
are applied directly to the image array for a more photographic look.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


OUTPUT_DIR = Path(__file__).resolve().parent
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
POINT_COUNT = 5_000
IMAGE_SIZE = 800
GRAIN_AMOUNT = 0.035
SEED = 12


def fibonacci_surface(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.arange(n, dtype=float) + 0.5
    y = 1.0 - 2.0 * i / n
    ring = np.sqrt(1.0 - y * y)
    theta = GOLDEN_ANGLE * i
    x = ring * np.cos(theta)
    z = ring * np.sin(theta)
    return x, y, z


def add_splat(
    image: np.ndarray,
    px: float,
    py: float,
    radius: float,
    strength: float,
) -> None:
    """Paint one Gaussian blob into an image array."""
    extent = max(1, int(np.ceil(radius * 3.0)))
    center_x = int(round(px))
    center_y = int(round(py))
    left = max(0, center_x - extent)
    right = min(image.shape[1], center_x + extent + 1)
    top = max(0, center_y - extent)
    bottom = min(image.shape[0], center_y + extent + 1)

    if left >= right or top >= bottom:
        return

    yy, xx = np.mgrid[top:bottom, left:right]
    distance_squared = (xx - px) ** 2 + (yy - py) ** 2
    image[top:bottom, left:right] += strength * np.exp(
        -distance_squared / (2.0 * max(radius * radius, 0.08))
    )


def main() -> None:
    x, y, z = fibonacci_surface(POINT_COUNT)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    sphere_radius = IMAGE_SIZE / (2.0 * 1.05)
    center = IMAGE_SIZE * 0.5
    front = (z + 1.0) * 0.5

    for point_x, point_y, point_front in zip(x, y, front):
        screen_x = center + point_x * sphere_radius
        screen_y = center - point_y * sphere_radius
        add_splat(
            canvas,
            screen_x,
            screen_y,
            radius=0.35 + 1.5 * point_front,
            strength=0.08 + 0.85 * point_front,
        )

    image = 1.0 - np.exp(-canvas * 0.9)
    image = np.clip(image, 0.0, 1.0)

    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    radial_distance = np.sqrt(
        ((xx - center) / sphere_radius) ** 2
        + ((yy - center) / sphere_radius) ** 2
    )
    sphere_mask = (radial_distance <= 1.0).astype(np.float32)
    vignette = np.clip(1.10 - 0.45 * radial_distance**2, 0.0, 1.0)
    vignette *= sphere_mask

    rng = np.random.default_rng(SEED)
    grain = rng.normal(0.0, GRAIN_AMOUNT, image.shape)
    image = np.clip((image + grain) * vignette, 0.0, 1.0)

    blue = np.array(to_rgb("tab:blue"), dtype=np.float32)
    rgb = 1.0 - image[..., None] * (1.0 - blue)

    figure, axis = plt.subplots(figsize=(7, 7), facecolor="white")
    axis.imshow(rgb, extent=(-1.05, 1.05, -1.05, 1.05), origin="upper")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_title("Fibonacci sphere with grain and vignette")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True, alpha=0.2)
    output_path = OUTPUT_DIR / "grain_vignette.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
