# Point Shapes and Particle Spheres

Small NumPy/Matplotlib experiments that build shapes from equations, Fibonacci
point distributions, and rasterized particles. Every script saves its image in
this folder.

Run a script from the project root with:

```bash
MPLBACKEND=Agg python shapes/fib_spiral.py
```

## Scripts, from basic to advanced

1. **`fib_spiral.py` — Fibonacci disk**

   Creates a sunflower-like pattern. The square-root radius keeps the points
   evenly distributed across the disk.

   ```python
   radius = np.sqrt(index / POINT_COUNT)
   angle = index * GOLDEN_ANGLE
   x, y = radius * np.cos(angle), radius * np.sin(angle)
   ```

   Output: `fib_spiral.png`

2. **`shapes.py` — Equation-based curves**

   Draws a selected shape from a parameter `t`. Set `SHAPE` to `circle`,
   `rose`, `lissajous`, `spiral`, `cardioid`, or `lemniscate`.

   ```python
   t = np.linspace(0, 2 * np.pi, 1_000)
   x, y = lemniscate(t)  # selected by SHAPE
   plt.scatter(x, y, color="tab:blue")
   ```

   Output: `<SHAPE>.png`

3. **`uniform_points.py` — Uniform random disk points**

   Compares a naive random radius with the correct area-preserving radius.
   `sqrt(random())` prevents the center from becoming too dense.

   ```python
   naive_radius = rng.random(POINT_COUNT)
   uniform_radius = np.sqrt(naive_radius)
   ```

   Output: `uniform_points.png`

4. **`fib_spiral_sphere.py` — Fibonacci sphere projection**

   Places points on a unit sphere, then displays only its `x`/`y` projection.
   The `z` coordinate is calculated but not used for visual depth.

   ```python
   y = 1.0 - 2.0 * (index + 0.5) / POINT_COUNT
   ring = np.sqrt(1.0 - y * y)
   x, z = ring * np.cos(theta), ring * np.sin(theta)
   ```

   Output: `fib_spiral_sphere.png`

5. **`depth_points.py` — Depth as particle size**

   Extends the sphere projection by treating `z` as front-to-back depth.
   Points nearer the viewer become larger.

   ```python
   front = (z + 1.0) * 0.5
   point_size = 2.0 + 18.0 * front
   axis.scatter(x, y, s=point_size, color="tab:blue")
   ```

   Output: `depth_points.png`

6. **`warp_points.py` — Warped sphere distribution**

   Changes the sphere latitude before calculating its ring radius. Values of
   `p` below `1` concentrate points near the poles; values above `1` move them
   toward the equator. Edit `P_VALUES` to compare settings.

   ```python
   y = np.sign(base_y) * np.abs(base_y) ** p
   ring = np.sqrt(1.0 - y * y)
   ```

   Output: `warp_points.png`

7. **`splat_points.py` — Soft raster particles**

   Replaces sharp scatter markers with Gaussian blobs painted directly into a
   floating-point image array. Overlapping blobs are compressed for smoother
   highlights.

   ```python
   image[top:bottom, left:right] += strength * np.exp(
       -distance_squared / (2.0 * sigma_squared)
   )
   ```

   Output: `splat_points.png`

8. **`grain_vignette.py` — Textured sphere rendering**

   Builds on raster splats by adding random grain and a radial vignette, then
   blends the result with `tab:blue` for a softer photographic appearance.

   ```python
   grain = rng.normal(0.0, GRAIN_AMOUNT, image.shape)
   image = np.clip((image + grain) * vignette, 0.0, 1.0)
   rgb = 1.0 - image[..., None] * (1.0 - blue)
   ```

   Output: `grain_vignette.png`

9. **`particle_sphere_warp.py` — Combined particle sphere**

   Combines the pole warp from `warp_points.py` with depth-sized main particles
   and a smaller random dust layer. Use `--pole-bias` to control the shape:
   `0.25` is strongly pole-focused, while `1.0` is uniform.

   ```python
   front = (z + 1.0) * 0.5
   point_size = 2.0 + 18.0 * front
   axis.scatter(x, y, s=point_size, color="tab:blue", alpha=0.82)
   ```

   Output: `particle-sphere-warp.png`

   Example:

   ```bash
   MPLBACKEND=Agg python shapes/particle_sphere_warp.py --pole-bias 0.5
   ```

## Common ideas

- `GOLDEN_ANGLE` spreads points without obvious radial lines.
- `tab:blue` keeps point plots visually consistent.
- `OUTPUT_DIR = Path(__file__).resolve().parent` makes outputs stay beside
  their scripts, regardless of the current working directory.
- Increase `POINT_COUNT` for denser plots; reduce it for faster experiments.
