#!/usr/bin/env python3
"""Render a grainy particle sphere to PNG and/or an animated GIF.

The sphere is made from points on its surface rather than points filling its
volume.  It includes a dark equatorial gap, bright polar clusters, depth-based
shading, dust, soft splats, and a subtle vignette.

Dependencies:
    pip install numpy pillow        # or on Termux:  pkg install python-numpy python-pillow

Example:
    python particle_sphere.py                       # PNG only
    python particle_sphere.py -o sphere.png --size 1200 --seed 7
    python particle_sphere.py --gif                 # PNG + animated GIF
    python particle_sphere.py --gif --gif-frames 60 # 60-frame GIF
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter
except ImportError as exc:  # pragma: no cover - useful message on a new Termux install
    missing = "numpy and Pillow"
    print(
        f"Missing {missing}. Install them in Termux with:\n"
        "  pkg install python\n"
        "  pip install numpy pillow",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def fibonacci_surface(count: int, gap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evenly distributed points on a unit sphere, excluding the gap."""
    index = np.arange(count, dtype=np.float64) + 0.5
    y = 1.0 - 2.0 * index / count
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = GOLDEN_ANGLE * index
    x = ring * np.cos(theta)
    z = ring * np.sin(theta)

    keep = np.abs(y) >= gap
    return x[keep], y[keep], z[keep]


def random_surface(
    rng: np.random.Generator, count: int, gap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return random surface points, useful for the fine dusty texture."""
    y = rng.uniform(-1.0, 1.0, count)
    theta = rng.uniform(0.0, 2.0 * math.pi, count)
    ring = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    keep = np.abs(y) >= gap
    return ring[keep] * np.cos(theta[keep]), y[keep], ring[keep] * np.sin(theta[keep])


def polar_cluster(
    rng: np.random.Generator, count: int, cap_angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return points concentrated near both poles and a per-point intensity."""
    # sqrt() spreads points over the cap instead of putting them all at its edge.
    lat = cap_angle * np.sqrt(rng.random(count))
    theta = rng.uniform(0.0, 2.0 * math.pi, count)
    tangent = np.sin(lat)
    around = tangent * np.cos(theta)
    depth = tangent * np.sin(theta)
    pole = np.where(rng.random(count) < 0.5, -1.0, 1.0)

    x = around
    y = pole * np.cos(lat)
    z = depth
    intensity = 1.05 + 0.55 * rng.random(count)
    return x, y, z, intensity


def rotate_y(x: np.ndarray, y: np.ndarray, z: np.ndarray, angle: float):
    """Rotate around the vertical axis, preserving the horizontal gap."""
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated_x = cosine * x + sine * z
    rotated_z = -sine * x + cosine * z
    return rotated_x, y, rotated_z


def add_splat(
    image: np.ndarray,
    px: float,
    py: float,
    radius: float,
    strength: float,
) -> None:
    """Add one small Gaussian point to a floating-point image buffer."""
    if strength <= 0.0 or radius <= 0.0:
        return

    extent = max(1, int(math.ceil(radius * 3.0)))
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
    sigma_squared = max(radius * radius, 0.08)
    image[top:bottom, left:right] += strength * np.exp(
        -distance_squared / (2.0 * sigma_squared)
    )


def render(args: argparse.Namespace, rotation_degrees: float | None = None) -> Image.Image:
    """Render a single frame of the particle sphere.

    When ``rotation_degrees`` is given, use it instead of ``args.rotation``;
    this is how the animated GIF produces many frames from a stable set of
    surface points and grain.  A separate RNG stream is used for the grain so
    the noise pattern does not flicker between animation frames.
    """
    rng = np.random.default_rng(args.seed)
    # Independent stream for image grain: keeps noise stable across animation frames.
    grain_rng = np.random.default_rng(args.seed + 0x5A5A5A5A)
    rotation = math.radians(
        rotation_degrees if rotation_degrees is not None else args.rotation
    )
    size = args.size
    height = int(round(size * args.aspect))
    height = max(64, height)

    # Keep the sphere circular even when a portrait/landscape aspect is requested.
    sphere_radius = min(size, height) * args.radius
    center_x = size * 0.5 + args.offset_x * size
    center_y = height * 0.5 + args.offset_y * height

    # Main, evenly distributed surface points.
    x, y, z = fibonacci_surface(args.points, args.gap)
    x, y, z = rotate_y(x, y, z, rotation)
    front = np.clip((z + 1.0) * 0.5, 0.0, 1.0)
    # Back-facing points remain visible but are quieter, giving a translucent dust ball.
    brightness = 0.10 + 0.90 * np.power(front, 1.45)
    point_radius = 0.34 + 1.55 * np.power(front, 1.15)

    # Fine random points add an irregular, grainy layer without filling the sphere.
    dust_x, dust_y, dust_z = random_surface(rng, args.dust, args.gap)
    dust_x, dust_y, dust_z = rotate_y(
        dust_x, dust_y, dust_z, rotation
    )
    dust_front = np.clip((dust_z + 1.0) * 0.5, 0.0, 1.0)
    dust_brightness = (0.035 + 0.23 * dust_front) * rng.uniform(
        0.55, 1.35, len(dust_x)
    )
    dust_radius = rng.uniform(0.16, 0.62, len(dust_x))

    # Bright clusters at the top and bottom poles.
    pole_x, pole_y, pole_z, pole_strength = polar_cluster(
        rng, args.poles, math.radians(args.pole_size)
    )
    pole_x, pole_y, pole_z = rotate_y(
        pole_x, pole_y, pole_z, rotation
    )
    pole_front = np.clip((pole_z + 1.0) * 0.5, 0.0, 1.0)
    pole_brightness = pole_strength * (0.55 + 0.75 * pole_front)
    pole_radius = rng.uniform(0.45, 1.65, len(pole_x)) * (0.75 + pole_front)

    core = np.zeros((height, size), dtype=np.float32)
    glow_source = np.zeros_like(core)

    def draw_points(
        points_x: np.ndarray,
        points_y: np.ndarray,
        strengths: np.ndarray,
        radii: np.ndarray,
        target: np.ndarray,
        glow_target: np.ndarray | None = None,
    ) -> None:
        for point_x, point_y, strength, radius in zip(
            points_x, points_y, strengths, radii
        ):
            screen_x = center_x + point_x * sphere_radius
            screen_y = center_y - point_y * sphere_radius
            add_splat(target, screen_x, screen_y, float(radius), float(strength))
            if glow_target is not None and strength > 0.48:
                add_splat(
                    glow_target,
                    screen_x,
                    screen_y,
                    float(radius) * 1.8,
                    float(strength) * 0.16,
                )

    draw_points(x, y, brightness, point_radius, core, glow_source)
    draw_points(dust_x, dust_y, dust_brightness, dust_radius, core)
    draw_points(pole_x, pole_y, pole_brightness, pole_radius, core, glow_source)

    # Add a soft halo around the brightest points.
    glow = np.asarray(
        Image.fromarray(np.uint8(np.clip(glow_source * 255.0, 0, 255)), mode="L")
        .filter(ImageFilter.GaussianBlur(max(1.0, size / 220.0))),
        dtype=np.float32,
    ) / 255.0

    # A soft circular mask keeps noise and glow inside the sphere.  The edge is
    # feathered, so the silhouette does not look like a hard vector circle.
    yy, xx = np.mgrid[0:height, 0:size]
    normalized_x = (xx - center_x) / sphere_radius
    normalized_y = (yy - center_y) / sphere_radius
    radial_distance = np.sqrt(normalized_x * normalized_x + normalized_y * normalized_y)
    sphere_mask = np.clip((1.0 - radial_distance) / 0.035, 0.0, 1.0)
    inside = radial_distance <= 1.0
    sphere_mask *= inside
    vignette = np.clip(1.12 - 0.42 * radial_distance * radial_distance, 0.0, 1.0)

    # Low-level sensor/grain noise gives the result a dusty photographic texture.
    # Uses grain_rng so the pattern is identical across animation frames.
    grain = grain_rng.normal(0.0, args.grain, (height, size)).astype(np.float32)
    grain *= sphere_mask * (0.25 + 0.75 * grain_rng.random((height, size)))

    # Slightly brighten the central body while retaining the dark equatorial gap.
    result = (core * 0.82 + glow * 0.72 + grain) * vignette
    result = np.clip(result, 0.0, 1.0)

    # Save as grayscale; convert to RGB for broad image-viewer compatibility.
    pixels = np.uint8(np.round(result * 255.0))
    return Image.fromarray(pixels, mode="L").convert("RGB")


def render_animation(args: argparse.Namespace) -> list[Image.Image]:
    """Render a list of frames that together form one full rotation.

    Each frame shares the same surface points and grain noise; only the
    rotation around the vertical axis advances by ``360 / args.gif_frames``
    degrees.  The GIF frames use ``args.gif_size`` so the resulting file
    stays a reasonable size.
    """
    gif_args = argparse.Namespace(**vars(args))
    gif_args.size = args.gif_size
    step = 360.0 / args.gif_frames
    return [
        render(gif_args, args.rotation + step * index)
        for index in range(args.gif_frames)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("particle-sphere.png"))
    parser.add_argument("--size", type=int, default=900, help="Image width in pixels (default: 900)")
    parser.add_argument("--aspect", type=float, default=1.0, help="Height / width (default: 1.0)")
    parser.add_argument("--points", type=int, default=5200, help="Main surface points")
    parser.add_argument("--dust", type=int, default=2800, help="Fine dust points")
    parser.add_argument("--poles", type=int, default=850, help="Extra points near the poles")
    parser.add_argument("--gap", type=float, default=0.035, help="Half-width of the dark equatorial gap")
    parser.add_argument("--pole-size", type=float, default=18.0, help="Pole cap angle in degrees")
    parser.add_argument("--radius", type=float, default=0.36, help="Sphere radius as a fraction of the short edge")
    parser.add_argument("--rotation", type=float, default=28.0, help="Rotation around vertical axis in degrees")
    parser.add_argument("--offset-x", type=float, default=0.0, help="Horizontal sphere offset as image fraction")
    parser.add_argument("--offset-y", type=float, default=0.0, help="Vertical sphere offset as image fraction")
    parser.add_argument("--grain", type=float, default=0.018, help="Amount of fine image grain")
    parser.add_argument("--seed", type=int, default=12, help="Random seed for repeatable images")
    parser.add_argument("--gif", action="store_true", help="Also produce an animated GIF that rotates the sphere a full turn")
    parser.add_argument("--no-png", action="store_true", help="With --gif, skip writing the PNG (GIF only)")
    parser.add_argument("--gif-output", type=Path, default=Path("particle-sphere.gif"), help="Animated GIF output path (default: particle-sphere.gif)")
    parser.add_argument("--gif-frames", type=int, default=36, help="Number of frames in the GIF; one full rotation is split across this many frames (default: 36)")
    parser.add_argument("--gif-fps", type=float, default=20.0, help="Playback speed of the GIF in frames per second (default: 20)")
    parser.add_argument("--gif-size", type=int, default=400, help="Frame width for the GIF, in pixels (default: 400)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.size < 64:
        raise SystemExit("--size must be at least 64")
    if not 0.1 <= args.aspect <= 4.0:
        raise SystemExit("--aspect must be between 0.1 and 4.0")
    if not 0.05 <= args.radius <= 0.49:
        raise SystemExit("--radius must be between 0.05 and 0.49")
    if args.gif and args.gif_frames < 2:
        raise SystemExit("--gif-frames must be at least 2")
    if args.gif and args.gif_fps <= 0:
        raise SystemExit("--gif-fps must be positive")
    if args.gif and args.gif_size < 64:
        raise SystemExit("--gif-size must be at least 64")

    if args.gif:
        # Always render the full-size image first so the PNG matches --size.
        if not args.no_png:
            image = render(args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            image.save(args.output)
            print(f"Saved {args.output} ({image.width}x{image.height})")
        # Then render the smaller frames for the animated GIF.
        frames = render_animation(args)
        args.gif_output.parent.mkdir(parents=True, exist_ok=True)
        first = frames[0]
        duration_ms = max(1, int(round(1000.0 / args.gif_fps)))
        first.save(
            args.gif_output,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(
            f"Saved {args.gif_output} "
            f"({len(frames)} frames @ {args.gif_fps:.0f}fps, "
            f"{first.width}x{first.height})"
        )
    else:
        image = render(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)
        print(f"Saved {args.output} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
