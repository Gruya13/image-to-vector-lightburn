"""Centralna konfiguracija, ogranicenja i parametri obrade."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("IMG2VEC_MODELS_DIR", PROJECT_ROOT / "models"))
PRESETS_DIR = Path(os.environ.get("IMG2VEC_PRESETS_DIR", PROJECT_ROOT / "presets"))
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

# Sistemska zastita (plan: 10 MB, 2000 px po duzoj strani)
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_LONG_SIDE_PX = 2000
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# Rezolucija lasera 0.1 mm - minimalna relevantna dimenzija detalja
LASER_RESOLUTION_MM = 0.1

DEFAULT_BG_MODEL = "isnet-general-use"
BG_MODELS = {
    "isnet-general-use": "Brzi (ISNet general)",
    "birefnet-general": "Maksimalni kvalitet (BiRefNet)",
}

Mode = Literal["stencil", "centerline", "mesh", "wireframe"]
MODE_LABELS = {
    "stencil": "Puna silueta (Fill)",
    "centerline": "Jednostruke linije (Line)",
    "mesh": "2D low-poly mreza",
    "wireframe": "3D wireframe iluzija",
}


@dataclass
class ProcessParams:
    """Svi parametri jedne obrade (serijalizuju se u JSON preset)."""

    # --- Izlaz i rezim -----------------------------------------------------
    mode: Mode = "stencil"
    target_width_mm: float | None = 150.0
    target_height_mm: float | None = None
    quality: int = 3  # 1..3 utice na gustinu mreze / broj izolinija

    # --- Pozadina i maska --------------------------------------------------
    remove_background: bool = True
    bg_model: str = DEFAULT_BG_MODEL
    invert: bool = False
    max_long_side_px: int = MAX_LONG_SIDE_PX
    min_island_area_px: int = 30

    # --- Stencil (potrace) -------------------------------------------------
    # "auto" bira Otsu za bimodalne (silueta/logo/tekst) i adaptive za foto
    threshold_method: Literal["adaptive", "otsu", "auto"] = "auto"
    threshold_block: int = 31
    threshold_c: float = 10.0
    smooth_radius: int = 2
    turd_size: int = 2
    opt_tolerance: float = 0.2

    # --- Centerline (skelet) ----------------------------------------------
    # "auto" bira XDoG za fotografske i threshold za line-art (bimodalne) slike
    line_threshold_method: Literal["auto", "xdog", "adaptive", "otsu"] = "auto"
    xdog_sigma: float = 0.8
    xdog_k: float = 1.6
    xdog_epsilon: float = 0.025
    min_spur_px: int = 12
    min_path_length_mm: float = 1.0
    merge_tolerance_mm: float = 0.2
    simplify_tolerance_mm: float = 0.05

    # --- Cut obris ---------------------------------------------------------
    add_cut: bool = False
    cut_offset_mm: float = 3.0
    cut_simplify_mm: float = 0.05
    cut_min_area_px: int = 40
    cut_buffer_resolution: int = 16

    # --- Faza 2 (2D low-poly mesh) ----------------------------------------
    mesh_spacing_mm: float = 3.0
    mesh_edge_threshold: float = 0.62
    mesh_max_edge_mm: float = 14.0

    # --- Faza 3 (3D wireframe) --------------------------------------------
    depth_levels: int = 18
    depth_parallax: float = 0.35
    depth_smoothing: int = 9

    # --- Opste -------------------------------------------------------------
    jpeg_quality: int = 92

    # -- validacija / serijalizacija ---------------------------------------
    def validate(self) -> None:
        if self.mode not in MODE_LABELS:
            raise ValueError(f"Nepoznat rezim: {self.mode!r}")
        if not (self.target_width_mm or self.target_height_mm):
            raise ValueError("Zadata sirina ili visina u mm je obavezna")
        for name in ("target_width_mm", "target_height_mm"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} mora biti pozitivan broj")
        if self.threshold_block < 3 or self.threshold_block % 2 == 0:
            raise ValueError("threshold_block mora biti neparan broj >= 3")
        if self.threshold_method not in ("adaptive", "otsu", "auto"):
            raise ValueError(f"Nepoznat threshold_method: {self.threshold_method!r}")
        if self.line_threshold_method not in ("auto", "xdog", "adaptive", "otsu"):
            raise ValueError(
                f"Nepoznat line_threshold_method: {self.line_threshold_method!r}")
        if not (0 <= self.cut_offset_mm <= 50):
            raise ValueError("cut_offset_mm mora biti u opsegu 0..50")
        if self.min_island_area_px < 0:
            raise ValueError("min_island_area_px ne moze biti negativan")
        if not (1 <= self.quality <= 3):
            raise ValueError("quality mora biti 1, 2 ili 3")
        if not (1 <= self.depth_levels <= 80):
            raise ValueError("depth_levels mora biti u opsegu 1..80")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessParams":
        """Tolerantno ucitavanje iz JSON-a (nepoznati kljucevi se ignorisu)."""
        known = {f.name for f in dataclasses.fields(cls)}
        clean: dict[str, Any] = {}
        for key, value in (data or {}).items():
            if key not in known:
                continue
            field_info = next(f for f in dataclasses.fields(cls) if f.name == key)
            annotation = str(field_info.type)
            if value is None and "None" in annotation:
                clean[key] = None
            elif "bool" in annotation:
                clean[key] = bool(value)
            elif "int" in annotation and "float" not in annotation:
                clean[key] = int(value)
            elif "float" in annotation or "None" in annotation:
                clean[key] = float(value)
            else:
                clean[key] = value
        params = cls(**clean)
        params.validate()
        return params


@dataclass
class RenderSettings:
    """Neserijalizovana podesavanja renderovanja (za API/CLI)."""

    include_dxf: bool = True
    write_mask_png: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))