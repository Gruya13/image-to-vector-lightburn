"""Pomocne funkcije za testove: sinteticke slike u memoriji."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def png_bytes(array: np.ndarray) -> bytes:
    """Kodira numpy niz (HxW ili HxWx3/4) u PNG bajtove."""
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    elif array.shape[2] == 3:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    elif array.shape[2] == 4:
        image = Image.fromarray(array.astype(np.uint8), mode="RGBA")
    else:  # pragma: no cover
        raise ValueError("Neocekivan oblik niza")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def white_canvas(width: int = 400, height: int = 400) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def black_rectangle(width: int = 400, height: int = 400,
                    x0: int = 100, y0: int = 100,
                    x1: int = 300, y1: int = 300) -> bytes:
    canvas = white_canvas(width, height)
    canvas[y0:y1, x0:x1] = 0
    return png_bytes(canvas)


def black_circle(size: int = 400, radius: int = 120,
                 center: tuple[int, int] | None = None) -> bytes:
    canvas = white_canvas(size, size)
    cy, cx = center or (size // 2, size // 2)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[mask] = 0
    return png_bytes(canvas)


def horizontal_line(width: int = 400, height: int = 120,
                    thickness: int = 8) -> bytes:
    canvas = white_canvas(width, height)
    y0 = height // 2 - thickness // 2
    canvas[y0:y0 + thickness, 40:width - 40] = 0
    return png_bytes(canvas)


def alpha_circle(size: int = 300, radius: int = 90) -> bytes:
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    cy = cx = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[inside, :3] = 20
    canvas[inside, 3] = 255
    return png_bytes(canvas)
