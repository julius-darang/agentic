"""Scatter shapes whose points come from equations.

Run with:
    python shapes.py

Change SHAPE below to try another equation-based shape. Available shapes are
circle, rose, lissajous, spiral, cardioid, and lemniscate. Nothing is placed
by hand: every point is calculated from t.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent


# Try: "circle", "rose", "lissajous", "spiral", "cardioid", or "lemniscate".
SHAPE = "lemniscate"


def circle(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A circle: x = cos(t), y = sin(t)."""
    x = np.cos(t)
    y = np.sin(t)
    return x, y


def rose(t: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A rose curve: r = cos(k*t), converted from polar to Cartesian."""
    r = np.cos(k * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def lissajous(
    t: np.ndarray, a: int = 3, b: int = 2, delta: float = np.pi / 2
) -> tuple[np.ndarray, np.ndarray]:
    """A Lissajous curve using x = sin(a*t + delta), y = sin(b*t)."""
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    return x, y


def spiral(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An Archimedean spiral: r = 0.15*t."""
    r = 0.15 * t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def cardioid(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A cardioid: r = 1 - cos(t), converted to Cartesian coordinates."""
    r = 1 - np.cos(t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def lemniscate(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A lemniscate (figure eight) using a rational parametrization."""
    denominator = 1 + np.sin(t) ** 2
    x = np.cos(t) / denominator
    y = np.sin(t) * np.cos(t) / denominator
    return x, y


def main() -> None:
    # t is the parameter. More values create a denser scatter of points.
    t = np.linspace(0, 2 * np.pi, 1_000)

    if SHAPE == "circle":
        x, y = circle(t)
        title = "Circle from x = cos(t), y = sin(t)"
    elif SHAPE == "rose":
        x, y = rose(t, k=5)
        title = "Rose curve from r = cos(5t)"
    elif SHAPE == "lissajous":
        x, y = lissajous(t, a=3, b=2)
        title = "Lissajous curve"
    elif SHAPE == "spiral":
        x, y = spiral(t)
        title = "Archimedean spiral"
    elif SHAPE == "cardioid":
        x, y = cardioid(t)
        title = "Cardioid from r = 1 - cos(t)"
    elif SHAPE == "lemniscate":
        x, y = lemniscate(t)
        title = "Lemniscate"
    else:
        raise ValueError(f"Unknown shape: {SHAPE}")

    # The shape is rendered entirely from the calculated x and y arrays.
    plt.scatter(x, y, s=8, color="tab:blue", alpha=0.75)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)

    output_path = OUTPUT_DIR / f"{SHAPE}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
