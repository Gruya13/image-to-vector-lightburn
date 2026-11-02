"""Orkestracija obrade: slika -> maska -> vektor -> mm -> Cut -> SVG/DXF."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .centerline import vectorize_centerline
from .config import LASER_RESOLUTION_MM, ProcessParams
from .cutout import cut_paths_from_mask
from .dxf_export import paths_to_dxf
from .geometry import (COLOR_CUT, COLOR_ENGRAVE, PathStats, VectorPath,
                       compute_stats, paths_bbox)
from .image_io import SourceImage, encode_png, load_image_bytes
from .optimize import despeckle_paths, sort_compound_groups
from .segment import SegmentationError, subject_mask
from .stencil import build_ink_mask, vectorize_stencil
from .svg_export import paths_to_svg


class PipelineError(RuntimeError):
    """Obrada slike nije moguca."""


@dataclass
class ProcessingResult:
    svg: str
    dxf: str
    paths: list[VectorPath]
    width_mm: float
    height_mm: float
    stats: PathStats
    mode: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    mask_png: bytes | None = None
    ink_png: bytes | None = None
    coverage: float | None = None
    source_size: tuple[int, int] = (0, 0)
    work_size: tuple[int, int] = (0, 0)

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "width_mm": round(self.width_mm, 3),
            "height_mm": round(self.height_mm, 3),
            "stats": self.stats.as_dict(),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "warnings": list(self.warnings),
            "coverage": None if self.coverage is None else round(self.coverage, 4),
            "source_size": list(self.source_size),
            "work_size": list(self.work_size),
            "version": __version__,
        }


def resolve_geometry(image: SourceImage, params: ProcessParams) -> tuple[float, float, float]:
    """Vraca (scale px->mm, sirina platna mm, visina platna mm)."""
    width_px, height_px = image.work_size
    if params.target_width_mm is not None:
        scale = params.target_width_mm / width_px
    elif params.target_height_mm is not None:
        scale = params.target_height_mm / height_px
    else:  # pragma: no cover - validate() ovo ne dozvoljava
        raise PipelineError("Zadata sirina ili visina u mm je obavezna")

    margin_mm = params.cut_offset_mm if params.add_cut else 0.0
    width_mm = width_px * scale + 2 * margin_mm
    height_mm = height_px * scale + 2 * margin_mm
    if width_mm <= 0 or height_mm <= 0:  # pragma: no cover
        raise PipelineError("Izracunate dimenzije platna nisu validne")
    return scale, width_mm, height_mm


def _prepare_mask(image: SourceImage, params: ProcessParams,
                  warnings: list[str]) -> np.ndarray | None:
    try:
        mask = subject_mask(
            image,
            remove_background=params.remove_background,
            bg_model=params.bg_model,
            invert=params.invert,
            min_island_area_px=params.min_island_area_px,
        )
    except SegmentationError as exc:
        warnings.append(f"Segmentacija nije uspela ({exc}); obrada ide bez maske")
        return None
    if mask is not None:
        coverage = float((mask > 127).mean())
        if coverage < 0.01:
            warnings.append(
                "Maska subjekta pokriva manje od 1% slike - proveri prag ili "
                "iskljuci uklanjanje pozadine")
        elif coverage > 0.98:
            warnings.append(
                "Maska pokriva skoro celu sliku - uklanjanje pozadine verovatno "
                "nije izdvojilo subjekt")
    return mask


def _finite_check(paths: list[VectorPath], warnings: list[str]) -> list[VectorPath]:
    valid: list[VectorPath] = []
    dropped = 0
    for path in paths:
        try:
            points = path.flatten(0.5)
        except ValueError:
            dropped += 1
            continue
        if all(np.isfinite([point.x, point.y]).all() for point in points):
            valid.append(path)
        else:
            dropped += 1
    if dropped:
        warnings.append(f"Odbačeno {dropped} nevalidnih putanja (NaN/inf koordinate)")
    return valid


def process_image(source: bytes | SourceImage | str | Path,
                  params: ProcessParams, *,
                  include_dxf: bool = True,
                  include_mask_png: bool = True,
                  mask_override: np.ndarray | None = None) -> ProcessingResult:
    """Glavna ulazna tacka Faze 1."""
    params.validate()
    warnings: list[str] = []
    timings: dict[str, float] = {}
    start = time.perf_counter()

    # --- 1. ucitavanje ------------------------------------------------------
    t = time.perf_counter()
    if isinstance(source, (bytes, bytearray)):
        image = load_image_bytes(bytes(source))
    elif isinstance(source, SourceImage):
        image = source
    else:
        path = Path(source)
        image = load_image_bytes(path.read_bytes(), filename=path.name,
                                 max_long_side=params.max_long_side_px)
    if image.work_size != image.original_size:
        warnings.append(
            f"Slika je smanjena sa {image.original_size[0]}x{image.original_size[1]} "
            f"na {image.work_size[0]}x{image.work_size[1]} px (max "
            f"{params.max_long_side_px} px)")
    timings["load"] = (time.perf_counter() - t) * 1000

    # --- 2. maska subjekta --------------------------------------------------
    t = time.perf_counter()
    if mask_override is not None:
        # rucno ociscena maska (cetkica/gumica u UI-ju) ima prednost
        mask = np.ascontiguousarray(mask_override)
        if mask.shape != (image.height, image.width):
            mask = cv2.resize(mask, (image.width, image.height),
                              interpolation=cv2.INTER_NEAREST)
        mask = ((mask > 127).astype(np.uint8) * 255)
    else:
        mask = _prepare_mask(image, params, warnings) if (
            params.remove_background or image.alpha is not None) else None
    coverage = float((mask > 127).mean()) if mask is not None else None
    timings["mask"] = (time.perf_counter() - t) * 1000

    # --- 3. geometrija platna ----------------------------------------------
    scale, width_mm, height_mm = resolve_geometry(image, params)
    offset_x = offset_y = params.cut_offset_mm if params.add_cut else 0.0

    # --- 4. vektorizacija ---------------------------------------------------
    t = time.perf_counter()
    engrave_paths: list[VectorPath] = []
    ink_png: bytes | None = None
    if params.mode == "stencil":
        px_paths = vectorize_stencil(image, mask, params)
        engrave_paths = [path.transformed(scale, offset_x, offset_y)
                         for path in px_paths if not path.is_empty()]
        # Fill putanje se ne sortiraju pojedinacno: compound grupe (spoljna
        # kontura + rupe) moraju ostati uzastopne za pravilan evenodd fill.
        engrave_paths = sort_compound_groups(engrave_paths)
        if include_mask_png:
            ink = build_ink_mask(image, mask, params)
            ink_png = encode_png(ink)
    elif params.mode == "centerline":
        engrave_paths = vectorize_centerline(
            image, mask, params, scale=scale,
            offset_x=offset_x, offset_y=offset_y)
        engrave_paths = despeckle_paths(engrave_paths, params.min_path_length_mm)
    elif params.mode == "mesh":
        from .phase2 import vectorize_mesh
        engrave_paths = vectorize_mesh(image, mask, params, scale=scale,
                                       offset_x=offset_x, offset_y=offset_y)
    elif params.mode == "wireframe":
        from .phase3 import vectorize_wireframe
        engrave_paths = vectorize_wireframe(image, mask, params, scale=scale,
                                            offset_x=offset_x, offset_y=offset_y)
    else:  # pragma: no cover - validate() hvata
        raise PipelineError(f"Nepodrzan rezim: {params.mode}")
    timings["vectorize"] = (time.perf_counter() - t) * 1000

    # --- 5. Cut obris -------------------------------------------------------
    t = time.perf_counter()
    cut_paths: list[VectorPath] = []
    if params.add_cut:
        cut_mask = mask
        if cut_mask is None and params.mode == "stencil":
            cut_mask = build_ink_mask(image, None, params)
            warnings.append(
                "Cut obris je izveden iz binarne gravure jer maska subjekta nije "
                "dostupna (iskljuceno uklanjanje pozadine)")
        if cut_mask is None:
            warnings.append(
                "Cut obris je preskocen: potrebna je maska subjekta (ukljuci "
                "uklanjanje pozadine ili koristi PNG sa alfa kanalom)")
        else:
            cut_paths = cut_paths_from_mask(
                cut_mask, scale=scale, offset_mm=params.cut_offset_mm,
                simplify_mm=params.cut_simplify_mm,
                min_area_px=params.cut_min_area_px,
                offset_x=offset_x, offset_y=offset_y,
                buffer_resolution=params.cut_buffer_resolution)
            if not cut_paths:
                warnings.append("Cut obris nije pronadjen u maski")
    timings["cut"] = (time.perf_counter() - t) * 1000

    # --- 6. finalizacija ----------------------------------------------------
    all_paths = _finite_check(engrave_paths + cut_paths, warnings)
    if not engrave_paths:
        warnings.append(
            "Nijedna gravura nije izdvojena - probaj drugi rezim ili parametre "
            "pragova")
    if engrave_paths:
        content_bbox = paths_bbox(engrave_paths)
        if content_bbox is not None:
            tolerance = max(0.5, LASER_RESOLUTION_MM * 5)
            if (content_bbox.min_x < -tolerance or content_bbox.min_y < -tolerance
                    or content_bbox.max_x > width_mm + tolerance
                    or content_bbox.max_y > height_mm + tolerance):
                warnings.append("Sadrzaj izlazi van platna - proveri dimenzije i offset")

    stats = compute_stats(all_paths, width_mm, height_mm)
    t = time.perf_counter()
    title = image.filename or "img2vec"
    svg = paths_to_svg(all_paths, width_mm, height_mm, title=title)
    dxf = paths_to_dxf(all_paths, width_mm, height_mm, title=title) if include_dxf else ""
    timings["export"] = (time.perf_counter() - t) * 1000
    timings["total"] = (time.perf_counter() - start) * 1000

    return ProcessingResult(
        svg=svg,
        dxf=dxf,
        paths=all_paths,
        width_mm=width_mm,
        height_mm=height_mm,
        stats=stats,
        mode=params.mode,
        timings_ms=timings,
        warnings=warnings,
        mask_png=encode_png(mask) if (include_mask_png and mask is not None) else None,
        ink_png=ink_png,
        coverage=coverage,
        source_size=image.original_size,
        work_size=image.work_size,
    )