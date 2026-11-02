"""Deljeni Material Preseti (ugradjeni + korisnicki JSON u presets/)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PRESETS_DIR, ProcessParams

_PRESET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "led-akril": {
        "name": "LED Akrilna Lampa",
        "description": "Centerline gravura za osvetljeni akril + spoljni cut offset 3 mm",
        "params": {
            "mode": "centerline",
            "remove_background": False,
            "line_threshold_method": "xdog",
            "min_path_length_mm": 1.0,
            "target_width_mm": 150.0,
            "add_cut": True,
            "cut_offset_mm": 3.0,
        },
    },
    "drvo-portret": {
        "name": "Gravura Drvo / Sperploca",
        "description": "Puna silueta sa filtriranim sitnim detaljima (bez progorevanja)",
        "params": {
            "mode": "stencil",
            "remove_background": True,
            "min_island_area_px": 60,
            "turdsize": 8,
            "opt_tolerance": 0.3,
            "target_width_mm": 150.0,
            "add_cut": False,
        },
    },
    "stencil-kontura": {
        "name": "Stencil / Kontura za Secenje",
        "description": "Zatvorena uproscena kontura + crveni Cut offset 3 mm",
        "params": {
            "mode": "stencil",
            "remove_background": True,
            "threshold_method": "otsu",
            "min_island_area_px": 80,
            "turdsize": 10,
            "opt_tolerance": 0.4,
            "target_width_mm": 120.0,
            "add_cut": True,
            "cut_offset_mm": 3.0,
        },
    },
    "line-art": {
        "name": "Line Art (Crtez)",
        "description": "Skeletonizacija crteza olovkom/tusem bez AI maske",
        "params": {
            "mode": "centerline",
            "remove_background": False,
            "line_threshold_method": "otsu",
            "min_spur_px": 14,
            "min_path_length_mm": 1.5,
            "target_width_mm": 150.0,
            "add_cut": False,
        },
    },
}


@dataclass
class Preset:
    key: str
    name: str
    description: str
    params: ProcessParams
    builtin: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "params": self.params.to_dict(),
            "builtin": self.builtin,
        }


def _preset_path(key: str) -> Path:
    return PRESETS_DIR / f"{key}.json"


def _read_preset_file(path: Path) -> Preset | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        params = ProcessParams.from_dict(data.get("params", {}))
    except (OSError, ValueError):
        return None
    return Preset(
        key=path.stem,
        name=str(data.get("name") or path.stem),
        description=str(data.get("description") or ""),
        params=params,
        builtin=False,
    )


def list_presets() -> list[Preset]:
    presets: dict[str, Preset] = {}
    for key, entry in BUILTIN_PRESETS.items():
        presets[key] = Preset(
            key=key,
            name=entry["name"],
            description=entry["description"],
            params=ProcessParams.from_dict(entry["params"]),
            builtin=True,
        )
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(PRESETS_DIR.glob("*.json")):
        preset = _read_preset_file(path)
        if preset is not None:
            presets[preset.key] = preset
    return list(presets.values())


def get_preset(key: str) -> Preset | None:
    for preset in list_presets():
        if preset.key == key:
            return preset
    return None


def save_preset(key: str, name: str, description: str,
                params: ProcessParams) -> Preset:
    key = key.strip().lower()
    if not _PRESET_NAME_RE.match(key):
        raise ValueError(
            "Naziv preseta moze sadrzati samo mala slova, cifre, '-' i '_' "
            "(2-64 znaka)")
    if key in BUILTIN_PRESETS:
        raise ValueError(f"'{key}' je ugradjeni preset i ne moze se prepisati")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name.strip() or key,
        "description": description.strip(),
        "params": params.to_dict(),
    }
    _preset_path(key).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return Preset(key=key, name=payload["name"], description=payload["description"],
                  params=params, builtin=False)


def delete_preset(key: str) -> bool:
    if key in BUILTIN_PRESETS:
        raise ValueError(f"'{key}' je ugradjeni preset i ne moze se obrisati")
    path = _preset_path(key)
    if path.is_file():
        path.unlink()
        return True
    return False