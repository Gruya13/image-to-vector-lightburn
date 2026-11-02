"""Ucitavanje i normalizacija ulaznih slika (limiti, alfa, downscale)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_INPUT_BYTES, MAX_LONG_SIDE_PX, SUPPORTED_EXTENSIONS


class ImageLoadError(ValueError):
    """Slika ne moze da se ucita ili ne zadovoljava ogranicenja."""


@dataclass
class SourceImage:
    """Normalizovana ulazna slika u radnoj rezoluciji."""

    rgb: np.ndarray            # HxWx3 uint8 (alfa kompozitovana na belo)
    alpha: np.ndarray | None   # HxW uint8 ili None
    original_size: tuple[int, int]
    work_size: tuple[int, int]
    filename: str = ""

    @property
    def width(self) -> int:
        return self.work_size[0]

    @property
    def height(self) -> int:
        return self.work_size[1]

    @property
    def work_scale(self) -> float:
        """work_px / original_px (<= 1 kad je slika smanjena)."""
        return self.work_size[0] / self.original_size[0]

    def composition_on_white(self) -> np.ndarray:
        return self.rgb


def _extract_alpha(image: Image.Image) -> np.ndarray | None:
    if image.mode in ("RGBA", "LA"):
        return np.asarray(image.getchannel("A"), dtype=np.uint8)
    if image.mode == "P":
        transparency = image.info.get("transparency")
        if transparency is not None:
            rgba = image.convert("RGBA")
            return np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    return None


def _composite_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def load_image_bytes(data: bytes, *, filename: str = "",
                     max_bytes: int = MAX_INPUT_BYTES,
                     max_long_side: int = MAX_LONG_SIDE_PX) -> SourceImage:
    """Ucitava sliku iz bajtova uz sva sistemska ogranicenja."""
    if not data:
        raise ImageLoadError("Prazan fajl")
    if len(data) > max_bytes:
        raise ImageLoadError(
            f"Fajl je prevelik: {len(data) / 1024 / 1024:.2f} MB "
            f"(maksimum {max_bytes / 1024 / 1024:.0f} MB)")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"Format slike nije prepoznat: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    original_size = image.size
    alpha = _extract_alpha(image)
    rgb_image = _composite_on_white(image)

    longest = max(original_size)
    scale = min(1.0, max_long_side / float(longest))
    if scale < 1.0:
        new_size = (max(1, round(original_size[0] * scale)),
                    max(1, round(original_size[1] * scale)))
        rgb_image = rgb_image.resize(new_size, Image.LANCZOS)
        if alpha is not None:
            alpha = np.asarray(
                Image.fromarray(alpha, mode="L").resize(new_size, Image.LANCZOS),
                dtype=np.uint8)

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:  # pragma: no cover - odbrana
        raise ImageLoadError("Neocekivan broj kanala nakon konverzije")
    return SourceImage(
        rgb=rgb,
        alpha=alpha,
        original_size=original_size,
        work_size=(rgb.shape[1], rgb.shape[0]),
        filename=filename,
    )


def load_image_file(path: str | Path, **kwargs) -> SourceImage:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Nepodrzan format: {file_path.suffix}")
    return load_image_bytes(file_path.read_bytes(), filename=file_path.name, **kwargs)


def encode_png(array: np.ndarray) -> bytes:
    """Kodira HxW ili HxWx3 uint8 niz u PNG (za mask preview)."""
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L")
    elif array.ndim == 3:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    else:  # pragma: no cover
        raise ValueError("Neocekivan oblik niza za PNG")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
