"""Rezim A: Stencil / Silhouette - binarna maska pa potrace Bezier krive."""

from __future__ import annotations

import cv2
import numpy as np

from .config import ProcessParams
from .geometry import STYLE_ENGRAVE_FILL, VectorPath
from .image_io import SourceImage
from .potrace import PotraceOptions, PotraceError, trace_bitmap, potrace_available


def _bilateral(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return gray
    diameter = max(5, int(radius) * 4 + 1)
    return cv2.bilateralFilter(gray, diameter, sigmaColor=75, sigmaSpace=75)


def histogram_bimodality(gray: np.ndarray, subject: np.ndarray | None = None
                         ) -> tuple[int, float, float]:
    """Vraca (otsu_prag, eta_separabilnost, udeo_tamne_klase).

    Eta je Otsu separabilnost (0..1): 1 znaci savrseno dve klase. Fotografije sa
    vise tonova imaju nisku etu, dok siluete/logo/tekst/line-art imaju visoku.
    """
    mask = ((subject > 127).astype(np.uint8) * 255) if subject is not None else None
    histogram = cv2.calcHist([gray], [0], mask, [256], [0, 256]).ravel().astype(np.float64)
    total = float(histogram.sum())
    if total <= 0:
        return 128, 0.0, 0.0
    probabilities = histogram / total
    bins = np.arange(256, dtype=np.float64)
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * bins)
    mu_total = float(mu[-1])
    denominator = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_between = np.where(
            denominator > 0,
            (mu_total * omega - mu) ** 2 / denominator,
            0.0)
    threshold = int(np.argmax(sigma_between))
    between = float(sigma_between[threshold])
    total_variance = float(np.sum(probabilities * (bins - mu_total) ** 2))
    eta = between / total_variance if total_variance > 1e-12 else 0.0
    return threshold, eta, float(omega[threshold])


def is_bimodal(gray: np.ndarray, subject: np.ndarray | None = None,
               *, eta_min: float = 0.86) -> bool:
    """Heuristika: da li je slika prakticno crno-bela (silueta/logo/tekst/line art).

    Bimodalne slike zahtevaju globalni (Otsu) prag, jer adaptive threshold ne
    popunjava velike jednobojne povrsine. Fotografije traze adaptive prag.
    """
    _threshold, eta, dark_share = histogram_bimodality(gray, subject)
    return eta >= eta_min and 0.002 < dark_share < 0.995


def resolve_threshold_method(gray: np.ndarray, subject: np.ndarray | None,
                             requested: str) -> str:
    if requested != "auto":
        return requested
    return "otsu" if is_bimodal(gray, subject) else "adaptive"


def build_ink_mask(image: SourceImage, subject: np.ndarray | None,
                   params: ProcessParams) -> np.ndarray:
    """Binarizuje sliku u masku gde je 255 = gravura (ink)."""
    gray = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)
    if subject is not None:
        # van subjekta je bela pozadina, da pozadina ne udje u gravuru
        gray = np.where(subject > 127, gray, 255).astype(np.uint8)
    if params.invert:
        gray = 255 - gray
    gray = _bilateral(gray, params.smooth_radius)

    method = resolve_threshold_method(gray, subject, params.threshold_method)
    block = max(3, int(params.threshold_block) | 1)
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, float(params.threshold_c))
    else:
        _threshold, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if params.smooth_radius >= 2:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= max(1, params.min_island_area_px):
            cleaned[labels == label] = 255
    return cleaned


def vectorize_stencil(image: SourceImage, subject: np.ndarray | None,
                      params: ProcessParams) -> list[VectorPath]:
    """Binarizuje i vektorizuje preko potrace; koordinate ostaju u pikselima."""
    if not potrace_available():
        raise PotraceError(
            "Stencil rezim zahteva potrace. Pokreni 'python scripts/setup_potrace.py' "
            "ili instaliraj potrace preko sistemskog paket menadzera.")
    ink = build_ink_mask(image, subject, params)
    if not ink.any():
        return []
    return trace_bitmap(
        ink > 0,
        options=PotraceOptions(turdsize=params.turd_size,
                               opttolerance=params.opt_tolerance),
        style=STYLE_ENGRAVE_FILL,
    )