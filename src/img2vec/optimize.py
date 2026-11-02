"""Post-procesiranje putanja: despeckle, spajanje, uproscavanje i sortiranje.

Napomena o vpype verziji: vpype 1.15 je refaktorisan i bibliotecki API vise ne
izlaze funkcije linesimplify/linesort (ostale su CLI komande). Zato se spajanje
krajeva radi preko vpype LineCollection.merge() (linemerge) za male skupove,
odnosno prostornim hashom za velike skupove (vpype merge je O(n^2)). Uproscavanje
(Ramer-Douglas-Peucker) i sortiranje (nearest-neighbour) su implementirani ovde,
sa identicnom semantikom i tolerancijama u milimetrima.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import vpype as vp

from .geometry import Point, VectorPath, path_from_coords

VPYPE_MERGE_LIMIT = 400


def despeckle_paths(paths: Sequence[VectorPath], min_length_mm: float, *,
                    keep_closed: bool = True) -> list[VectorPath]:
    """Brise mikro-putanje krace od praga (da laser ne pravi tackaste opekotine)."""
    kept: list[VectorPath] = []
    for path in paths:
        if path.is_empty():
            continue
        if path.closed and keep_closed:
            kept.append(path)
            continue
        if path.length() >= min_length_mm:
            kept.append(path)
    return kept


# ---------------------------------------------------------------------------
# Ramer-Douglas-Peucker uproscavanje polilinija
# ---------------------------------------------------------------------------

def simplify_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Vektorizovan Ramer-Douglas-Peucker za (N, 2) niz tacaka."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or tolerance <= 0:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start, end = pts[first], pts[last]
        delta = end - start
        norm = float(np.hypot(delta[0], delta[1]))
        segment = pts[first + 1:last]
        if norm < 1e-12:
            distances = np.hypot(segment[:, 0] - start[0], segment[:, 1] - start[1])
        else:
            distances = np.abs(
                delta[1] * (segment[:, 0] - start[0])
                - delta[0] * (segment[:, 1] - start[1])) / norm
        max_index = int(np.argmax(distances))
        if distances[max_index] > tolerance:
            split = first + 1 + max_index
            keep[split] = True
            stack.append((first, split))
            stack.append((split, last))
    return pts[keep]


def simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Cuva zatvorenost polilinije (prva tacka = poslednja) pri uproscavanju."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts
    closed = bool(np.allclose(pts[0], pts[-1]))
    if closed:
        body = simplify_points(pts[:-1], tolerance)
        return np.vstack([body, body[0:1]])
    return simplify_points(pts, tolerance)


# ---------------------------------------------------------------------------
# Spajanje krajeva: vpype linemerge (mali skupovi) / prostorni hash (veliki)
# ---------------------------------------------------------------------------

def _endpoint_key(point: np.ndarray, tolerance: float) -> tuple[int, int]:
    scale = 1.0 / max(tolerance, 1e-9)
    return (int(round(point[0] * scale)), int(round(point[1] * scale)))


def merge_endpoints(lines: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    """Spaja polilinije kojima se krajevi poklapaju unutar tolerancije."""
    arrays = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 2]
    if len(arrays) < 2:
        return arrays
    tolerance = max(float(tolerance), 1e-6)

    used = [False] * len(arrays)
    index: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def register(line_index: int, end: int) -> None:
        point = arrays[line_index][0 if end == 0 else -1]
        index.setdefault(_endpoint_key(point, tolerance), []).append((line_index, end))

    for i in range(len(arrays)):
        register(i, 0)
        register(i, 1)

    def find_next(point: np.ndarray) -> tuple[int, int] | None:
        base = _endpoint_key(point, tolerance)
        best: tuple[float, int, int] | None = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for line_index, end in index.get((base[0] + dx, base[1] + dy), ()):
                    if used[line_index]:
                        continue
                    end_point = arrays[line_index][0 if end == 0 else -1]
                    distance = float(np.hypot(*(end_point - point)))
                    if distance <= tolerance and (best is None or distance < best[0]):
                        best = (distance, line_index, end)
        if best is None:
            return None
        return best[1], best[2]

    merged: list[np.ndarray] = []
    for start_index in range(len(arrays)):
        if used[start_index]:
            continue
        used[start_index] = True
        parts: list[np.ndarray] = [arrays[start_index]]
        while True:
            candidate = find_next(parts[-1][-1])
            if candidate is None:
                break
            line_index, end = candidate
            used[line_index] = True
            parts.append(arrays[line_index] if end == 0 else arrays[line_index][::-1])
        while True:
            candidate = find_next(parts[0][0])
            if candidate is None:
                break
            line_index, end = candidate
            used[line_index] = True
            parts.insert(0, arrays[line_index] if end == 1 else arrays[line_index][::-1])
        merged.append(np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])
    return merged


def merge_polylines(lines: Iterable[np.ndarray], tolerance: float) -> list[np.ndarray]:
    """Spaja linije: vpype linemerge za male skupove, prostorni hash za velike."""
    arrays = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 2]
    if len(arrays) < 2:
        return arrays
    if len(arrays) <= VPYPE_MERGE_LIMIT:
        collection = vp.LineCollection()
        for array in arrays:
            collection.append(array[:, 0] + 1j * array[:, 1])
        collection.merge(tolerance, flip=True)
        return [np.column_stack([line.real, line.imag]) for line in collection.lines]
    return merge_endpoints(arrays, tolerance)


# ---------------------------------------------------------------------------
# Sortiranje nearest-neighbour (minimalan prazan hod)
# ---------------------------------------------------------------------------

def sort_polylines_nearest(lines: Iterable[np.ndarray],
                           origin: tuple[float, float] = (0.0, 0.0),
                           *, allow_reverse: bool = True) -> list[np.ndarray]:
    """Greedy nearest-neighbour redosled polilinija preko prostornog grida."""
    arrays = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 2]
    count = len(arrays)
    if count < 2:
        return arrays

    lengths = np.array([np.hypot(*(array[-1] - array[0])) + len(array)
                        for array in arrays])
    cell = max(1.0, float(np.median(lengths)))

    def grid_key(point: np.ndarray) -> tuple[int, int]:
        return (int(point[0] // cell), int(point[1] // cell))

    start_grid: dict[tuple[int, int], list[int]] = {}
    end_grid: dict[tuple[int, int], list[int]] = {}
    for index, array in enumerate(arrays):
        start_grid.setdefault(grid_key(array[0]), []).append(index)
        end_grid.setdefault(grid_key(array[-1]), []).append(index)

    used = [False] * count
    ordered: list[np.ndarray] = []
    current = np.array(origin, dtype=np.float64)

    def find_candidate() -> tuple[float, int, bool] | None:
        base = grid_key(current)
        best: tuple[float, int, bool] | None = None
        radius = 1
        while radius <= 64:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    for index in start_grid.get((base[0] + dx, base[1] + dy), ()):
                        if used[index]:
                            continue
                        distance = float(np.hypot(*(arrays[index][0] - current)))
                        if best is None or distance < best[0]:
                            best = (distance, index, False)
                    if allow_reverse:
                        for index in end_grid.get((base[0] + dx, base[1] + dy), ()):
                            if used[index]:
                                continue
                            distance = float(np.hypot(*(arrays[index][-1] - current)))
                            if best is None or distance < best[0]:
                                best = (distance, index, True)
            if best is not None:
                return best
            radius += 1
        for index in range(count):
            if used[index]:
                continue
            distance = float(np.hypot(*(arrays[index][0] - current)))
            if best is None or distance < best[0]:
                best = (distance, index, False)
            if allow_reverse:
                end_distance = float(np.hypot(*(arrays[index][-1] - current)))
                if end_distance < best[0]:
                    best = (end_distance, index, True)
        return best

    for _ in range(count):
        candidate = find_candidate()
        if candidate is None:  # pragma: no cover
            break
        _distance, index, reverse = candidate
        used[index] = True
        line = arrays[index][::-1].copy() if reverse else arrays[index]
        ordered.append(line)
        current = line[-1]
    return ordered


def sort_paths_nearest(paths: Sequence[VectorPath],
                       origin: Point = Point(0.0, 0.0),
                       *, allow_reverse: bool = True) -> list[VectorPath]:
    """Ista logika nad VectorPath objektima (za Bezier putanje stencil moda)."""
    remaining = [path for path in paths if not path.is_empty()]
    ordered: list[VectorPath] = []
    current = origin
    while remaining:
        best_index = 0
        best_distance = float("inf")
        best_reverse = False
        for index, path in enumerate(remaining):
            start_distance = current.distance_to(path.start)
            if start_distance < best_distance:
                best_distance = start_distance
                best_index = index
                best_reverse = False
            if allow_reverse and not path.closed:
                end_distance = current.distance_to(path.end)
                if end_distance < best_distance:
                    best_distance = end_distance
                    best_index = index
                    best_reverse = True
        path = remaining.pop(best_index)
        if best_reverse:
            path = path.reversed()
        ordered.append(path)
        current = path.end
    return ordered


def sort_compound_groups(paths: Sequence[VectorPath],
                         origin: Point = Point(0.0, 0.0)) -> list[VectorPath]:
    """Sortira grupe compound putanja (rupe ostaju uz svoju spoljnu konturu)."""
    groups: list[list[VectorPath]] = []
    for path in paths:
        compound_id = path.meta.get("compound")
        if (compound_id is not None and groups
                and groups[-1][0].meta.get("compound") == compound_id):
            groups[-1].append(path)
        else:
            groups.append([path])

    def group_start(group: list[VectorPath]) -> Point:
        return group[0].start

    remaining = groups
    ordered: list[VectorPath] = []
    current = origin
    while remaining:
        best_index = 0
        best_distance = float("inf")
        for index, group in enumerate(remaining):
            distance = current.distance_to(group_start(group))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        group = remaining.pop(best_index)
        ordered.extend(group)
        current = group[-1].end
    return ordered


# ---------------------------------------------------------------------------
# Glavni optimizacioni prolaz
# ---------------------------------------------------------------------------

def optimize_polylines(lines: Iterable[np.ndarray], *,
                       merge_tolerance_mm: float,
                       simplify_tolerance_mm: float,
                       sort: bool = True) -> list[np.ndarray]:
    """Spaja, uproscava i (opciono) sortira polilinije u milimetrima."""
    arrays = [np.asarray(line, dtype=np.float64) for line in lines if len(line) >= 2]
    if not arrays:
        return []
    merged = merge_polylines(arrays, merge_tolerance_mm)

    simplified: list[np.ndarray] = []
    for points in merged:
        result = simplify_polyline(points, simplify_tolerance_mm)
        if len(result) >= 2:
            simplified.append(result)
    if sort:
        simplified = sort_polylines_nearest(simplified)
    return simplified


def merge_and_simplify(paths: Sequence[VectorPath], *, merge_tolerance_mm: float,
                       simplify_tolerance_mm: float,
                       style=None) -> list[VectorPath]:
    """Spaja stroke putanje i vraca ih kao VectorPath.

    Putanje moraju biti stroke (centerline) tipa; Bezier krive se prvo
    flatten-uju na 0.01 mm (ispod rezolucije lasera).
    """
    lines = [np.array([(point.x, point.y) for point in path.flatten(0.01)],
                      dtype=np.float64)
             for path in paths if not path.is_empty()]
    styles = [path.style for path in paths if not path.is_empty()]
    optimized = optimize_polylines(
        lines, merge_tolerance_mm=merge_tolerance_mm,
        simplify_tolerance_mm=simplify_tolerance_mm)
    base_style = style or (styles[0] if styles else None)
    result: list[VectorPath] = []
    for line in optimized:
        vector = path_from_coords([(float(x), float(y)) for x, y in line],
                                  style=base_style)
        if not vector.is_empty():
            result.append(vector)
    return result