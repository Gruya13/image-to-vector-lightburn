"""Automatski spoljni Cut obris: maska -> mm poligon -> shapely offset."""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .geometry import STYLE_CUT, VectorPath, path_from_coords


def _largest_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry.geoms if not poly.is_empty]
    return [poly for poly in getattr(geometry, "geoms", []) if isinstance(poly, Polygon)]


def cut_paths_from_mask(mask: np.ndarray, *, scale: float, offset_mm: float,
                        simplify_mm: float = 0.05, min_area_px: int = 40,
                        offset_x: float = 0.0, offset_y: float = 0.0,
                        buffer_resolution: int = 16) -> list[VectorPath]:
    """Generise zatvorene Cut putanje kao offset spoljne konture maske.

    mask      - uint8 0..255 maska subjekta u radnoj rezoluciji
    scale     - px -> mm faktor
    offset_mm - spoljni odmak (0 = tacno po konturi)
    """
    binary = (np.asarray(mask) > 127).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    paths: list[VectorPath] = []
    for contour in contours:
        if cv2.contourArea(contour) < max(1, min_area_px):
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        mm_points = [(float(x) * scale + offset_x, float(y) * scale + offset_y)
                     for x, y in points]
        polygon = Polygon(mm_points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        # shapely: join_style=1 (round) -> glatki cut obris
        grown = polygon.buffer(offset_mm, join_style=1, quad_segs=buffer_resolution)
        for result in _largest_polygons(grown):
            simplified = result.simplify(simplify_mm, preserve_topology=True)
            for final in _largest_polygons(simplified):
                ring = [(float(x), float(y)) for x, y in final.exterior.coords]
                if len(ring) < 4:
                    continue
                paths.append(path_from_coords(ring, closed=True, style=STYLE_CUT))
    return paths