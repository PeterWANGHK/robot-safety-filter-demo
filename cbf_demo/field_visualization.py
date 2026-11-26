"""Display-only 3-D geometry for the computed Poisson safety field.

Marching cubes extracts constant-h surfaces from the existing scalar grid; it
does not rebuild the PDE, alter occupancy, or change the CBF constraints.
Vertices use the field's world coordinates in metres. The level values retain
the field units m^2 and must not be described as obstacle clearance.

Display cropping removes triangles unless all three vertices lie within the
requested box. It leaves open edges and is not an exact clipping algorithm.
The coarse extraction step and crop are visualization choices only, not
geometry for collision checking or a new safety certificate.
"""

from __future__ import annotations

import numpy as np


FIELD_LEVELS = (0.002, 0.008, 0.02)
FIELD_COLORS = ("#E69F00", "#56B4E9", "#0072B2")
DEFAULT_DISPLAY_BOUNDS = ((-0.60, -0.60, -0.03), (0.70, 0.60, 1.05))


def field_rgb(values) -> np.ndarray:
    """Map h in m^2 to a stable warm-orange to blue RGB display palette.

    Fixed anchor values avoid changing colors between scenes or extraction
    requests. Values outside the anchor range are clipped to the end colors.
    Invalid/occupied samples should be assigned an explicit color by callers.
    """
    values = np.asarray(values, dtype=float)
    anchors = np.array([
        [int(color[offset:offset + 2], 16) / 255 for offset in (1, 3, 5)]
        for color in FIELD_COLORS
    ])
    return np.stack([np.interp(values, FIELD_LEVELS, anchors[:, channel])
                     for channel in range(3)], axis=-1)


def _color_hex(level: float) -> str:
    rgb = np.rint(field_rgb(np.array(level)) * 255).astype(int)
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def build_field_meshes(field, levels=FIELD_LEVELS, bounds=None, step_size=2) -> list[dict]:
    """Extract display meshes as dictionaries of level/vertices/faces/color.

    ``field`` supplies ``h`` (NxMxK), ``origin`` (XYZ) and positive ``spacing``
    in metres. ``bounds`` is ``(lower_xyz, upper_xyz)`` in world metres, or
    None. Out-of-range levels, empty crops and constant grids yield no mesh for
    those levels. Levels are returned in increasing order with duplicates
    removed. The extraction dependency scikit-image is imported lazily.

    ``step_size`` is a positive integer grid stride for display extraction.
    Use step_size=1 for the least coarse representation; no extraction setting
    changes the CBF field used for planning/control.
    """
    values = np.asarray(field.h)
    origin = np.asarray(field.origin, dtype=float)
    spacing = float(field.spacing)
    if values.ndim != 3 or any(size < 2 for size in values.shape):
        raise ValueError("field.h must be a 3-D grid with at least two sites per axis")
    if not np.all(np.isfinite(values)):
        raise ValueError("field.h must contain finite scalar values")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("field.origin must be a finite XYZ vector")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("field.spacing must be finite and positive")
    if (not isinstance(step_size, (int, np.integer)) or isinstance(step_size, (bool, np.bool_))
            or step_size < 1):
        raise ValueError("step_size must be a positive integer")
    levels = np.asarray(tuple(levels), dtype=float)
    if levels.ndim != 1 or not np.all(np.isfinite(levels)):
        raise ValueError("levels must be a finite 1-D sequence")
    lower, upper = None, None
    if bounds is not None:
        corners = np.asarray(bounds, dtype=float)
        if corners.shape != (2, 3) or not np.all(np.isfinite(corners)) or np.any(corners[0] >= corners[1]):
            raise ValueError("bounds must contain finite, strictly increasing lower/upper XYZ corners")
        lower, upper = corners

    minimum, maximum = float(values.min()), float(values.max())
    requested = sorted(set(float(level) for level in levels if minimum < level < maximum))
    if not requested:
        return []
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError("3-D field extraction requires scikit-image; install the experiment's pinned dependencies") from exc

    meshes = []
    for level in requested:
        try:
            vertices, faces, _, _ = marching_cubes(
                values, level=level, spacing=(spacing, spacing, spacing),
                step_size=int(step_size), allow_degenerate=False,
            )
        except RuntimeError as exc:
            # Coarse strides can miss a small superlevel region even if the
            # full-resolution grid spans this level. Such a display is empty.
            if "No surface found" in str(exc):
                continue
            raise
        vertices = np.asarray(vertices, dtype=float) + origin
        faces = np.asarray(faces, dtype=np.int64)
        if lower is not None:
            inside = np.all(vertices >= lower, axis=1) & np.all(vertices <= upper, axis=1)
            faces = faces[np.all(inside[faces], axis=1)]
            if not len(faces):
                continue
            used, inverse = np.unique(faces.ravel(), return_inverse=True)
            vertices = vertices[used]
            faces = inverse.reshape(-1, 3)
        if len(faces):
            meshes.append(dict(level=level, vertices=vertices, faces=faces,
                               color=_color_hex(level)))
    return meshes
