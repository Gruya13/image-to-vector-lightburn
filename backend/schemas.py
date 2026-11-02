"""Pydantic modeli za Web API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RenderRequest(BaseModel):
    image_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class MaskRequest(BaseModel):
    """Rucno ociscena maska poslata kao PNG data URL (base64)."""

    mask_png: str


class PresetRequest(BaseModel):
    key: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=400)
    params: dict[str, Any] = Field(default_factory=dict)


class JobRequest(BaseModel):
    image_id: str
    params: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool
    version: str
    potrace: bool
    rembg_ready: bool
    depth_ready: bool
    providers: list[str]
    modes: dict[str, str]
    max_upload_mb: float


class RenderResponse(BaseModel):
    document_id: str
    svg_url: str
    width_mm: float
    height_mm: float
    stats: dict[str, Any]
    timings_ms: dict[str, float]
    warnings: list[str]
    mask_url: str | None = None
    ink_url: str | None = None
    coverage: float | None = None
    mode: str


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    progress: int = 0
    message: str = ""
    result: dict[str, Any] | None = None
