"""Brzi raster preview putanja (za Web UI i debug, bez spoljnih alata)."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import COLOR_CUT, COLOR_DETAIL, COLOR_ENGRAVE, VectorPath


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def render_preview(paths: list[VectorPath], width_mm: float, height_mm: float,
                   *, max_px: int = 1400, background: tuple[int, int, int] = (255, 255, 255),
                   engrave_color: str = COLOR_ENGRAVE,
                   detail_color: str = COLOR_DETAIL,
                   cut_color: str = COLOR_CUT,
                   layer_filter: set[str] | None = None) -> np.ndarray:
    """Rasterizuje putanje u BGR numpy sliku (bez LightBurn-a)."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Neispravne dimenzije za preview")
    scale = max_px / max(width_mm, height_mm)
    width_px = max(1, int(round(width_mm * scale)))
    height_px = max(1, int(round(height_mm * scale)))
    canvas = np.full((height_px, width_px, 3), background, dtype=np.uint8)

    fill_layers: list[tuple[list[VectorPath], str]] = []
    for path in paths:
        if path.is_empty():
            continue
        layer = path.style.layer_name
        if layer_filter is not None and layer not in layer_filter:
            continue
        if path.style.fill is not None:
            compound_id = path.meta.get("compound")
            if (compound_id is not None and fill_layers
                    and fill_layers[-1][0][0].meta.get("compound") == compound_id):
                fill_layers[-1][0].append(path)
            else:
                fill_layers.append(([path], path.style.fill))
            continue
        if path.style.stroke == cut_color:
            color = _hex_to_bgr(cut_color)
            thickness = max(1, int(round(0.25 * scale))) if scale > 4 else 1
        elif path.style.stroke == detail_color:
            color = _hex_to_bgr(detail_color)
            thickness = 1
        else:
            color = _hex_to_bgr(engrave_color)
            thickness = max(1, int(round(0.12 * scale))) if scale > 5 else 1
        points = _path_pixels(path, scale)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], path.closed, color, thickness,
                          lineType=cv2.LINE_AA)

    for group, fill in fill_layers:
        color = _hex_to_bgr(engrave_color if fill == COLOR_ENGRAVE else fill)
        # evenodd popunjavanje: XOR svih podputanja (spoljna kontura ^ rupe)
        mask = np.zeros((height_px, width_px), dtype=np.uint8)
        for path in group:
            points = _path_pixels(path, scale)
            if len(points) < 2:
                continue
            part = np.zeros_like(mask)
            cv2.fillPoly(part, [points], 255, lineType=cv2.LINE_AA)
            mask = cv2.bitwise_xor(mask, part)
        canvas[mask > 127] = color
    return canvas


def _path_pixels(path: VectorPath, scale: float) -> np.ndarray:
    return np.array([[int(round(p.x * scale)), int(round(p.y * scale))]
                     for p in path.flatten(0.05)], dtype=np.int32)


def encode_preview_png(paths: list[VectorPath], width_mm: float, height_mm: float,
                       **kwargs) -> bytes:
    canvas = render_preview(paths, width_mm, height_mm, **kwargs)
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG kodiranje preview-a nije uspelo")
    return buffer.tobytes()