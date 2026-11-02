"""Faza 3: 3D wireframe iluzija iz mape dubine (Depth Anything V2 + HLR).

Postupak:
  1. ONNX inferencija mape relativne dubine (model se preuzima u ./models).
  2. Bilateralno glađanje dubine da se smanji suma.
  3. Generisanje horizontalnih izolinija dubine i vertikalnih linija mreze.
  4. Pomeranje tacaka po Z osi (paralaksa dubine) -> 3D iluzija.
  5. Hidden Line Removal: z-buffer test vidljivosti i sečenje skrivenih delova.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR, ProcessParams
from .geometry import STYLE_ENGRAVE_LINE, VectorPath, path_from_coords
from .image_io import SourceImage
from .optimize import optimize_polylines

DEPTH_MODEL_FILE = "depth-anything-v2-small.onnx"
DEPTH_MODEL_URLS = [
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_int8.onnx",
]
MODEL_INPUT_SIZE = 518
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session_cache: dict[str, object] = {}


class DepthModelError(RuntimeError):
    """Model dubine nije dostupan."""


def depth_model_path() -> Path:
    return MODELS_DIR / DEPTH_MODEL_FILE


def ensure_depth_model(*, verbose: bool = False) -> Path:
    """Preuzima ONNX model dubine u ./models ako ga vec nema."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = depth_model_path()
    if target.is_file() and target.stat().st_size > 1_000_000:
        return target
    last_error: Exception | None = None
    partial = target.with_suffix(".onnx.part")
    for url in DEPTH_MODEL_URLS:
        try:
            if verbose:
                print(f"Preuzimanje modela dubine: {url}")
            urllib.request.urlretrieve(url, partial)  # noqa: S310
            partial.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            partial.unlink(missing_ok=True)
    raise DepthModelError(
        f"Model dubine nije dostupan ({last_error}). Pokreni "
        f"'python scripts/download_models.py --depth' ili postavi fajl u {target}."
    )


def _providers() -> list[str] | None:
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover
        return None
    ordered = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                           "CPUExecutionProvider") if p in available]
    return ordered or None


def _get_session() -> object:
    import onnxruntime as ort

    if "depth" in _session_cache:
        return _session_cache["depth"]
    model_path = ensure_depth_model()
    providers = _providers()
    if providers:
        session = ort.InferenceSession(str(model_path), providers=providers)
    else:  # pragma: no cover
        session = ort.InferenceSession(str(model_path))
    _session_cache["depth"] = session
    return session


def estimate_depth(rgb: np.ndarray) -> np.ndarray:
    """Vraca normalizovanu mapu dubine (H, W) u opsegu 0..1."""
    session = _get_session()
    height, width = rgb.shape[:2]
    resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - _MEAN) / _STD
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    inputs = session.get_inputs()
    input_name = inputs[0].name
    outputs = session.run(None, {input_name: tensor})
    depth = np.asarray(outputs[0])
    while depth.ndim > 2:
        depth = depth[0]
    depth = cv2.resize(depth.astype(np.float32), (width, height),
                       interpolation=cv2.INTER_CUBIC)
    return _normalize(depth)


def _normalize(array: np.ndarray) -> np.ndarray:
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum < 1e-9:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def _segment_by_mask(line: np.ndarray, mask: np.ndarray,
                     min_points: int = 3) -> list[np.ndarray]:
    """Deli poliliniju na neprekidne delove koji leze unutar maske."""
    if len(line) == 0:
        return []
    height, width = mask.shape
    xs = np.clip(np.round(line[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(line[:, 1]).astype(int), 0, height - 1)
    inside = mask[ys, xs].astype(bool)
    segments: list[np.ndarray] = []
    start: int | None = None
    for index, keep in enumerate(inside):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            if index - start >= min_points:
                segments.append(line[start:index])
            start = None
    if start is not None and len(inside) - start >= min_points:
        segments.append(line[start:])
    return segments


def _grid_lines(depth: np.ndarray, mask: np.ndarray, count: int) -> list[np.ndarray]:
    """Vertikalne linije mreze kroz dubinu (x = const)."""
    height, width = depth.shape
    lines: list[np.ndarray] = []
    step = max(4, width // max(2, count))
    for x in range(step // 2, width, step):
        column = np.nonzero(mask[:, x])[0]
        if len(column) < 3:
            continue
        y0, y1 = int(column[0]), int(column[-1])
        ys = np.arange(y0, y1 + 1, max(2, step // 2))
        if len(ys) < 3:
            continue
        column_line = np.column_stack([np.full(len(ys), x, dtype=np.float64),
                                       ys.astype(np.float64)])
        lines.extend(_segment_by_mask(column_line, mask, min_points=3))
    return lines


def _parallax(points: np.ndarray, depth: np.ndarray, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Pomera tacke po X i Y srazmerno dubini (3D iluzija) i vraca z vrednosti."""
    height, width = depth.shape
    xs = np.clip(points[:, 0].astype(int), 0, width - 1)
    ys = np.clip(points[:, 1].astype(int), 0, height - 1)
    z = depth[ys, xs]
    # paralaksa je ogranicena da sadrzaj ne izadje iz platna
    normalized = np.clip(spread * (z - 0.5), -0.16, 0.16)
    shift = normalized * width * 0.5
    shifted = points.copy()
    shifted[:, 0] += shift
    shifted[:, 1] += shift * 0.35  # blagi vertikalni pomeraj (ugao kamere)
    return shifted, z


def _rasterize_zbuffer(lines: list[tuple[np.ndarray, np.ndarray]],
                       height: int, width: int) -> np.ndarray:
    """Upisuje maksimalnu dubinu svake projektovane tacke u z-buffer."""
    z_buffer = np.full((height, width), -np.inf, dtype=np.float32)
    for projected, z_values in lines:
        for index in range(len(projected) - 1):
            x0, y0 = projected[index]
            x1, y1 = projected[index + 1]
            z0, z1 = z_values[index], z_values[index + 1]
            steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            xs = np.clip(np.round(np.linspace(x0, x1, steps)).astype(int), 0, width - 1)
            ys = np.clip(np.round(np.linspace(y0, y1, steps)).astype(int), 0, height - 1)
            zs = np.linspace(z0, z1, steps).astype(np.float32)
            np.maximum.at(z_buffer, (ys, xs), zs)
    return z_buffer


def hidden_line_removal(lines: list[tuple[np.ndarray, np.ndarray]],
                        height: int, width: int, *,
                        epsilon: float = 0.035) -> list[np.ndarray]:
    """Vraca samo vidljive delove linija (z-buffer test)."""
    z_buffer = _rasterize_zbuffer(lines, height, width)
    visible: list[np.ndarray] = []
    for projected, z_values in lines:
        xs = np.clip(np.round(projected[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.round(projected[:, 1]).astype(int), 0, height - 1)
        z_buffer_values = z_buffer[ys, xs]
        mask_visible = z_values >= (z_buffer_values - epsilon)

        # podeli na neprekidne vidljive segmente
        start: int | None = None
        for index, is_visible in enumerate(mask_visible):
            if is_visible and start is None:
                start = index
            elif not is_visible and start is not None:
                if index - start >= 2:
                    visible.append(projected[start:index])
                start = None
        if start is not None and len(mask_visible) - start >= 2:
            visible.append(projected[start:])
    return visible


def vectorize_wireframe(image: SourceImage, subject: np.ndarray | None,
                        params: ProcessParams, *, scale: float,
                        offset_x: float = 0.0,
                        offset_y: float = 0.0) -> list[VectorPath]:
    """3D wireframe iluzija sa HLR-om, izrazena u milimetrima."""
    try:
        depth = estimate_depth(image.rgb)
    except (DepthModelError, ImportError) as exc:
        raise DepthModelError(str(exc)) from exc

    smoothing = max(3, int(params.depth_smoothing) | 1)
    depth = cv2.bilateralFilter(depth, smoothing, 0.15, smoothing * 2.0)
    depth = _normalize(depth)

    if subject is not None:
        mask = (subject > 127).astype(np.uint8)
    else:
        # bez maske: koristi celu sliku, ali izbaci ivice (nema smisla)
        mask = np.ones(depth.shape, dtype=np.uint8)

    # dubina je nepouzdana na samoj granici objekta - erodiraj masku
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    if int(mask.sum()) < 100:
        raise DepthModelError("Maska je premala za 3D wireframe")

    levels = np.linspace(0.08, 0.92, max(3, int(params.depth_levels)))
    # horizontalne izolinije dubine (prate volumen) + vertikalna mreza
    candidate_lines = (_iso_lines(depth, mask, levels)
                       + _grid_lines(depth, mask, max(4, int(params.depth_levels))))
    if not candidate_lines:
        return []

    height, width = depth.shape
    # pomeranje tacaka po Z osi (paralaksa) prema dubini
    shifted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for line in candidate_lines:
        projected, z_values = _parallax(line, depth, params.depth_parallax)
        shifted_lines.append((projected, z_values))

    visible = hidden_line_removal(shifted_lines, height, width, epsilon=0.06)

    # px -> mm + optimizacija za laser (sa secenjem na platno zbog paralakse)
    width_mm = image.work_size[0] * scale + 2 * offset_x
    height_mm = image.work_size[1] * scale + 2 * offset_y
    lines_mm: list[np.ndarray] = []
    for line in visible:
        converted = line * scale
        converted[:, 0] += offset_x
        converted[:, 1] += offset_y
        lines_mm.extend(_clip_line(converted, width_mm, height_mm))
    optimized = optimize_polylines(
        lines_mm, merge_tolerance_mm=0.15, simplify_tolerance_mm=0.05, sort=True)

    paths: list[VectorPath] = []
    for line in optimized:
        path = path_from_coords([(float(x), float(y)) for x, y in line],
                                style=STYLE_ENGRAVE_LINE)
        if not path.is_empty() and path.length() >= params.min_path_length_mm:
            paths.append(path)
    return paths


def _clip_line(line: np.ndarray, width_mm: float,
               height_mm: float) -> list[np.ndarray]:
    """Sece poliliniju na pravougaonik platna (0..W, 0..H)."""
    from shapely.geometry import LineString, box

    if len(line) < 2:
        return []
    clipped = LineString(line).intersection(box(0, 0, width_mm, height_mm))
    if clipped.is_empty:
        return []
    if clipped.geom_type == "LineString":
        return [np.asarray(clipped.coords, dtype=np.float64)]
    return [np.asarray(part.coords, dtype=np.float64)
            for part in clipped.geoms if len(part.coords) >= 2]


def _iso_lines(depth: np.ndarray, mask: np.ndarray,
               levels: np.ndarray) -> list[np.ndarray]:
    """Izolinije dubine (izohipse), isecene na (erodiranu) masku objekta."""
    from skimage.measure import find_contours

    blur = cv2.GaussianBlur(depth, (0, 0), 1.2)
    lines: list[np.ndarray] = []
    for level in levels:
        for contour in find_contours(blur, float(level)):
            if len(contour) < 4:
                continue
            points = np.column_stack([contour[:, 1], contour[:, 0]])
            lines.extend(_segment_by_mask(points, mask, min_points=4))
    return lines