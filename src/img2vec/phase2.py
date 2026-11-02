"""Faza 2: 2D low-poly mesh (Delaunay triangulacija + deduplikacija ivica).

Postupak:
  1. Uzorkovanje tacaka unutar maske objekta - veca gustina na mestima visokog
     gradijenta ivica, manja na ravnim povrsinama.
  2. Delaunay triangulacija (scipy.spatial).
  3. Brisanje trouglova cija je teziste van maske (i previse dugih ivica).
  4. Deduplikacija ivica: set(tuple(sorted((i, j)))) uklanja duple linije
     izmedju susednih trouglova.
  5. Spajanje ivica u lance kroz vpype linemerge i skaliranje u mm.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import Delaunay

from .config import ProcessParams
from .geometry import STYLE_ENGRAVE_LINE, VectorPath, path_from_coords
from .image_io import SourceImage
from .optimize import optimize_polylines
from .stencil import build_ink_mask


def _shape_mask(image: SourceImage, subject: np.ndarray | None,
                params: ProcessParams) -> np.ndarray:
    if subject is not None:
        return (subject > 127).astype(np.uint8)
    return (build_ink_mask(image, None, params) > 127).astype(np.uint8)


def _edge_gradient(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    scale = float(np.percentile(magnitude, 99.0)) + 1e-9
    return np.clip(magnitude / scale, 0.0, 1.0)


def sample_points(mask: np.ndarray, gradient: np.ndarray, *,
                  spacing_px: int, edge_threshold: float,
                  seed: int = 7) -> np.ndarray:
    """Uzorkuje tacke unutar maske, gušće uz ivice i detalje."""
    height, width = mask.shape
    spacing = max(3, int(spacing_px))
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []

    ys = np.arange(spacing * 0.5, height, spacing)
    xs = np.arange(spacing * 0.5, width, spacing)
    for y in ys:
        for x in xs:
            xi, yi = int(min(width - 1, x)), int(min(height - 1, y))
            if not mask[yi, xi]:
                continue
            # gradijent pojacava gustinu tacaka (ivice/detalji)
            probability = 0.32 + 0.68 * float(gradient[yi, xi])
            if rng.random() > probability:
                continue
            jitter = spacing * 0.30
            px = float(np.clip(x + rng.uniform(-jitter, jitter), 0, width - 1))
            py = float(np.clip(y + rng.uniform(-jitter, jitter), 0, height - 1))
            if mask[int(py), int(px)]:
                points.append((px, py))

    # tacke duz konture (ostrija silueta)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 8:
            continue
        step = max(2, spacing // 2)
        for index in range(0, len(contour), step):
            x, y = contour[index, 0]
            points.append((float(x), float(y)))

    return np.asarray(points, dtype=np.float64)


def triangulate_edges(points: np.ndarray, mask: np.ndarray, *,
                      max_edge_px: float) -> list[tuple[int, int]]:
    """Delaunay -> filtriranje trouglova -> unikatne ivice."""
    if len(points) < 4:
        return []
    try:
        triangulation = Delaunay(points)
    except Exception:  # noqa: BLE001 - degenerisani skup tacaka
        return []

    height, width = mask.shape
    simplices = triangulation.simplices
    centroids = points[simplices].mean(axis=1)
    cx = np.clip(centroids[:, 0].astype(int), 0, width - 1)
    cy = np.clip(centroids[:, 1].astype(int), 0, height - 1)
    inside = mask[cy, cx].astype(bool)

    edges: set[tuple[int, int]] = set()
    for triangle, keep in zip(simplices, inside):
        if not keep:
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            index_a, index_b = int(triangle[a]), int(triangle[b])
            if (np.hypot(*(points[index_a] - points[index_b])) > max_edge_px):
                continue
            edges.add((min(index_a, index_b), max(index_a, index_b)))
    return sorted(edges)


def vectorize_mesh(image: SourceImage, subject: np.ndarray | None,
                   params: ProcessParams, *, scale: float,
                   offset_x: float = 0.0, offset_y: float = 0.0) -> list[VectorPath]:
    """Generise low-poly mrezu kao gravuru (crne linije) u milimetrima."""
    mask = _shape_mask(image, subject, params)
    gray = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)
    gradient = _edge_gradient(gray)

    spacing_px = max(3, int(round(params.mesh_spacing_mm / max(scale, 1e-6))))
    max_edge_px = max(6.0, params.mesh_max_edge_mm / max(scale, 1e-6))
    points = sample_points(mask, gradient, spacing_px=spacing_px,
                           edge_threshold=params.mesh_edge_threshold)
    if len(points) < 4:
        return []
    edges = triangulate_edges(points, mask, max_edge_px=max_edge_px)
    if not edges:
        return []

    # px -> mm i spajanje ivica kroz zajednicka temena (vpype linemerge)
    converted = points * scale
    converted[:, 0] += offset_x
    converted[:, 1] += offset_y
    lines = [np.array([converted[a], converted[b]], dtype=np.float64)
             for a, b in edges]
    optimized = optimize_polylines(
        lines, merge_tolerance_mm=0.05, simplify_tolerance_mm=0.0, sort=True)

    paths: list[VectorPath] = []
    for line in optimized:
        path = path_from_coords([(float(x), float(y)) for x, y in line],
                                style=STYLE_ENGRAVE_LINE)
        if not path.is_empty():
            paths.append(path)
    return paths
