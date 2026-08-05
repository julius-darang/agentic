#!/usr/bin/env python3
"""Render a warped Fibonacci particle sphere.

This combines the pole-focused distribution from ``warp_points.py`` with the
layered particle look of the sphere scripts.  The main particles use depth to
control their size, while a quieter dust layer adds texture without changing
the overall warped-sphere appearance.

Run with:
    python particle_sphere_warp.py

The default pole bias of 0.25 gives the output its characteristic concentration
near the north and south poles.  Use a value closer to 1 for a more uniform
sphere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
OUTPUT_PATH = Path(__file__).resolve().parent / "particle-sphere-warp.png"


def warped_surface(
    count: int, pole_bias: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Fibonacci points on a unit sphere warped toward its poles."""
    index = np.arange(count, dtype=float) + 0.5
    base_y = 1.0 - 2.0 * index / count
    y = np.sign(base_y) * np.abs(base_y) ** pole_bias
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = GOLDEN_ANGLE * index
    x = ring * np.cos(theta)
    z = ring * np.sin(theta)
    return x, y, z


def random_warped_surface(
    rng: np.random.Generator, count: int, pole_bias: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a random, pole-warped surface layer for fine particle dust."""
    base_y = rng.uniform(-1.0, 1.0, count)
    y = np.sign(base_y) * np.abs(base_y) ** pole_bias
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    return ring * np.cos(theta), y, ring * np.sin(theta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output image path (default: particle-sphere-warp.png)",
    )
    parser.add_argument(
        "--points", type=int, default=5_000, help="Number of main particles"
    )
    parser.add_argument(
        "--dust", type=int, default=2_000, help="Number of small dust particles"
    )
    parser.add_argument(
        "--pole-bias",
        type=float,
        default=0.25,
        help="Latitude warp: 0.25 is pole-focused, 1.0 is uniform",
    )
    parser.add_argument("--seed", type=int, default=12, help="Random seed for dust")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.points < 1:
        raise SystemExit("--points must be positive")
    if args.dust < 0:
        raise SystemExit("--dust must not be negative")
    if args.pole_bias <= 0:
        raise SystemExit("--pole-bias must be positive")

    x, y, z = warped_surface(args.points, args.pole_bias)
    front = (z + 1.0) * 0.5
    point_size = 2.0 + 18.0 * front

    rng = np.random.default_rng(args.seed)
    dust_x, dust_y, dust_z = random_warped_surface(
        rng, args.dust, args.pole_bias
    )
    dust_front = (dust_z + 1.0) * 0.5
    dust_size = 0.25 + 1.0 * dust_front

    # Draw farther points first so the larger front particles sit on top.
    order = np.argsort(z)

    figure, axis = plt.subplots(figsize=(7, 7), facecolor="white")
    if args.dust:
        axis.scatter(
            dust_x,
            dust_y,
            s=dust_size,
            color="tab:blue",
            alpha=0.22,
            linewidths=0,
        )
    axis.scatter(
        x[order],
        y[order],
        s=point_size[order],
        color="tab:blue",
        alpha=0.82,
        linewidths=0,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_title(f"Warped particle sphere (pole bias = {args.pole_bias:g})")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(True, alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
