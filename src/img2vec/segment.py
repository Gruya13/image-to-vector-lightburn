"""Segmentacija subjekta: alfa kanal, rembg (ONNX) i ciscenje maske."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DEFAULT_BG_MODEL, MODELS_DIR
from .image_io import SourceImage

_session_lock = threading.Lock()
_sessions: dict[str, Any] = {}
_last_error: dict[str, str] = {}


class SegmentationError(RuntimeError):
    """Uklanjanje pozadine nije uspelo."""


def _prepare_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))
    return MODELS_DIR


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime nije instaliran
        return []


def _preferred_providers() -> list[str] | None:
    """CUDA (Linux) -> CoreML (macOS) -> CPU, u zavisnosti sta je dostupno."""
    available = available_providers()
    if not available:
        return None
    ordered = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                           "CPUExecutionProvider") if p in available]
    return ordered or None


def get_session(model_name: str = DEFAULT_BG_MODEL) -> Any:
    """Vraca (i kesira) rembg sesiju sa preuzetim ONNX modelom."""
    _prepare_models_dir()
    with _session_lock:
        if model_name in _sessions:
            return _sessions[model_name]
        from rembg import new_session

        providers = _preferred_providers()
        try:
            if providers:
                session = new_session(model_name, providers=providers)
            else:  # pragma: no cover - bez onnxruntime providera
                session = new_session(model_name)
        except Exception as exc:  # noqa: BLE001 - rembg baca razne greske
            _last_error[model_name] = str(exc)
            raise SegmentationError(
                f"Model za uklanjanje pozadine '{model_name}' nije dostupan: {exc}"
            ) from exc
        _sessions[model_name] = session
        return session


def warmup(model_name: str = DEFAULT_BG_MODEL) -> bool:
    """Predgreva sesiju (poziva se pri startu servera, nikad u zahtevu)."""
    try:
        get_session(model_name)
        return True
    except SegmentationError:
        return False


def is_session_ready(model_name: str = DEFAULT_BG_MODEL) -> bool:
    return model_name in _sessions


def model_status() -> dict[str, Any]:
    return {
        "models_dir": str(MODELS_DIR),
        "sessions": sorted(_sessions),
        "providers": available_providers(),
        "errors": dict(_last_error),
    }


def _rembg_mask(rgb: np.ndarray, model_name: str) -> np.ndarray:
    from rembg import remove

    session = get_session(model_name)
    try:
        result = remove(rgb, session=session, only_mask=True, post_process_mask=True)
    except Exception as exc:  # noqa: BLE001
        raise SegmentationError(f"Uklanjanje pozadine nije uspelo: {exc}") from exc
    mask = np.asarray(result)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def clean_mask(mask: np.ndarray, min_island_area_px: int) -> np.ndarray:
    """Uklanja izolovana ostrva manja od praga (cv2 connectedComponents)."""
    binary = (np.asarray(mask) > 127).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary) * 255
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= max(1, min_island_area_px):
            output[labels == label] = 1
    return (output * 255).astype(np.uint8)


def subject_mask(image: SourceImage, *, remove_background: bool = True,
                 bg_model: str = DEFAULT_BG_MODEL, invert: bool = False,
                 min_island_area_px: int = 30) -> np.ndarray | None:
    """Vraca masku subjekta (0..255) ili None ako segmentacija nije trazena.

    Prioritet: postojeci alfa kanal -> rembg -> None (cela slika).
    """
    mask: np.ndarray | None = None
    if image.alpha is not None and bool((image.alpha < 250).any()):
        mask = image.alpha
    elif remove_background:
        # Za prakticno crno-bele slike (silueta/logo/tekst) pozadina je vec bela:
        # rembg je nepotreban i moze samo da unese sum u veze/ivice.
        from .stencil import is_bimodal
        gray = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)
        if is_bimodal(gray):
            return None
        mask = _rembg_mask(image.rgb, bg_model)
    if mask is None:
        return None
    if invert:
        mask = 255 - mask
    return clean_mask(mask, min_island_area_px)