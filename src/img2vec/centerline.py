"""Rezim B: Single-Line Art - XDoG/threshold -> skelet -> skan graf -> vpype."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .config import ProcessParams
from .geometry import STYLE_ENGRAVE_LINE, VectorPath, path_from_coords
from .image_io import SourceImage
from .optimize import optimize_polylines


def _bilateral(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return gray
    diameter = max(5, int(radius) * 4 + 1)
    return cv2.bilateralFilter(gray, diameter, sigmaColor=75, sigmaSpace=75)


def xdog_ink(gray: np.ndarray, params: ProcessParams) -> np.ndarray:
    """XDoG filter (Winnemoller): izdvaja pune poteze kao binarnu masku."""
    image = gray.astype(np.float32) / 255.0
    sigma = max(0.1, float(params.xdog_sigma))
    sigma2 = max(0.1, sigma * float(params.xdog_k))
    g1 = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma2)
    difference = g1 - g2  # tamni potezi daju negativnu razliku
    ink = difference < -abs(float(params.xdog_epsilon))
    return (ink.astype(np.uint8) * 255)


def build_line_mask(image: SourceImage, subject: np.ndarray | None,
                    params: ProcessParams) -> np.ndarray:
    """Vraca binarnu masku poteza (255 = potez)."""
    gray = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)
    if subject is not None:
        gray = np.where(subject > 127, gray, 255).astype(np.uint8)
    if params.invert:
        gray = 255 - gray

    method = params.line_threshold_method
    if method == "auto":
        from .stencil import is_bimodal
        # Line-art i skice su bimodalne -> threshold daje pune poteze (bez duplih
        # ivica koje XDoG proizvodi). Fotografije -> XDoG skica.
        method = "otsu" if is_bimodal(gray, subject) else "xdog"
    if method == "xdog":
        ink = xdog_ink(gray, params)
    elif method == "otsu":
        gray_for_threshold = _bilateral(gray, params.smooth_radius)
        _value, ink = cv2.threshold(
            gray_for_threshold, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        block = max(3, int(params.threshold_block) | 1)
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, float(params.threshold_c))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)

    # uklanjanje sitnog suma pre skeletonizacije
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(ink)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= max(1, params.min_island_area_px // 3):
            cleaned[labels == label] = 255
    return cleaned


def _frac_centerline(image: SourceImage, subject: np.ndarray | None,
                     params: ProcessParams) -> np.ndarray:
    ink = build_line_mask(image, subject, params)
    if not ink.any():
        return ink
    return (skeletonize(ink > 0).astype(np.uint8) * 255)


def _summarize_skeleton(skeleton_image: np.ndarray):
    from skan import Skeleton, summarize

    skeleton = Skeleton(skeleton_image.astype(np.uint8))
    summary = summarize(skeleton, separator="-")
    return skeleton, summary


def skeleton_to_polylines(skeleton_image: np.ndarray, min_spur_px: float) -> list[np.ndarray]:
    """Pretvara 1px skelet u polilinije (x, y) uz uklanjanje kratkih slepih grana.

    Koristi skan graf: cvorovi su junction/endpoint tacke, grane su lukovi.
    Slepе grane (junction->endpoint) krace od min_spur_px se odsecaju, a cvorovi
    stepena 2 se "provlace" kako bi se dobile sto duze neprekidne linije.
    """
    try:
        skeleton, summary = _summarize_skeleton(skeleton_image)
    except Exception:  # noqa: BLE001 - skan odbija degenerisane skeletе
        return []

    records = summary.to_dict("records")
    if not records:
        return []

    coords_cache: dict[int, np.ndarray] = {}

    def branch_coords(branch_id: int) -> np.ndarray:
        if branch_id not in coords_cache:
            coords = np.asarray(skeleton.path_coordinates(branch_id), dtype=np.float64)
            # skan vraca (row, col) -> (x, y)
            coords_cache[branch_id] = coords[:, ::-1].copy()
        return coords_cache[branch_id]

    edges: list[dict] = []
    # skan 0.13: path ID je redni broj reda u summarize() tabeli (kolona
    # "skeleton-id" je ID skeleta, ne grane), a path_coordinates prima taj ID.
    for path_id, record in enumerate(records):
        edges.append({
            "index": path_id,
            "src": int(record["node-id-src"]),
            "dst": int(record["node-id-dst"]),
            "type": int(record["branch-type"]),
            "distance": float(record["branch-distance"]),
            "coords": branch_coords(path_id),
        })

    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["src"]].append(edge["index"])
        adjacency[edge["dst"]].append(edge["index"])

    active = [True] * len(edges)

    def active_degree(node: int) -> int:
        return sum(1 for index in adjacency[node] if active[index])

    # 1) uklanjanje kratkih slepih grana (spur pruning), iterativno
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if not active[edge["index"]]:
                continue
            src_degree = active_degree(edge["src"])
            dst_degree = active_degree(edge["dst"])
            if edge["type"] == 1 and edge["distance"] < min_spur_px:
                if (src_degree == 1 and dst_degree >= 3) or (dst_degree == 1 and src_degree >= 3):
                    active[edge["index"]] = False
                    changed = True
            elif edge["type"] == 0 and edge["distance"] < min_spur_px:
                # izolovana linija bez cvora: brisi samo ako je kratka
                active[edge["index"]] = False
                changed = True

    # 2) obilazak lanca kroz cvorove stepena 2 (spajanje u duze polilinije)
    visited = [False] * len(edges)
    polylines: list[np.ndarray] = []

    def walk_chain(start_node: int, start_edge: int) -> np.ndarray | None:
        points: list[np.ndarray] = []
        node = start_node
        edge_index = start_edge
        while edge_index is not None:
            edge = edges[edge_index]
            if visited[edge_index]:
                break
            visited[edge_index] = True
            coords = edge["coords"]
            if edge["src"] == node:
                ordered = coords
                next_node = edge["dst"]
            else:
                ordered = coords[::-1]
                next_node = edge["src"]
            if points:
                points.append(ordered[1:])
            else:
                points.append(ordered)
            node = next_node
            if active_degree(node) != 2:
                break
            next_edge = next((index for index in adjacency[node]
                              if active[index] and not visited[index]), None)
            edge_index = next_edge
        if not points:
            return None
        return np.concatenate(points, axis=0)

    for edge in edges:
        if not active[edge["index"]]:
            continue
        for terminal in (edge["src"], edge["dst"]):
            if active_degree(terminal) != 2 and not visited[edge["index"]]:
                line = walk_chain(terminal, edge["index"])
                if line is not None and len(line) >= 2:
                    polylines.append(line)

    # 3) zatvoreni ciklusi (svi cvorovi stepena 2)
    for edge in edges:
        if not active[edge["index"]] or visited[edge["index"]]:
            continue
        line = walk_chain(edge["src"], edge["index"])
        if line is not None and len(line) >= 3:
            polylines.append(line)

    return polylines


def _polylines_to_mm(polylines: list[np.ndarray], scale: float,
                     offset_x: float, offset_y: float) -> list[np.ndarray]:
    converted = []
    for line in polylines:
        mm = line * scale
        mm[:, 0] += offset_x
        mm[:, 1] += offset_y
        converted.append(mm)
    return converted


def vectorize_centerline(image: SourceImage, subject: np.ndarray | None,
                         params: ProcessParams, *, scale: float,
                         offset_x: float = 0.0,
                         offset_y: float = 0.0) -> list[VectorPath]:
    """Skeletonizuje sliku i vraca optimizovane milimetarske linije."""
    skeleton_image = _frac_centerline(image, subject, params)
    if not skeleton_image.any():
        return []
    polylines_px = skeleton_to_polylines(skeleton_image, params.min_spur_px)
    if not polylines_px:
        return []
    polylines_mm = _polylines_to_mm(polylines_px, scale, offset_x, offset_y)

    # vpype linemerge + RDP uproscavanje + nearest-neighbour sortiranje
    optimized = optimize_polylines(
        polylines_mm,
        merge_tolerance_mm=params.merge_tolerance_mm,
        simplify_tolerance_mm=params.simplify_tolerance_mm,
        sort=True)

    paths: list[VectorPath] = []
    for line in optimized:
        path = path_from_coords([(float(x), float(y)) for x, y in line],
                                style=STYLE_ENGRAVE_LINE)
        if not path.is_empty() and path.length() >= params.min_path_length_mm:
            paths.append(path)
    return paths