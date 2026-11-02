"""FastAPI backend: upload, obrada, preview, izvoz, preseti i Faza 3 red cekanja."""

from __future__ import annotations

import base64
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from img2vec import __version__
from img2vec.config import MAX_INPUT_BYTES, MODE_LABELS, ProcessParams
from img2vec.image_io import ImageLoadError, SourceImage, encode_png, load_image_bytes
from img2vec.pipeline import PipelineError, ProcessingResult, process_image
from img2vec.potrace import find_potrace
from img2vec.preview import encode_preview_png
from img2vec.presets import delete_preset as delete_preset_file
from img2vec.presets import list_presets, save_preset
from img2vec.segment import available_providers, is_session_ready, warmup

from .schemas import (HealthResponse, JobRequest, JobResponse, MaskRequest,
                      PresetRequest, RenderRequest, RenderResponse)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_SESSIONS = 12
MAX_JOBS_KEPT = 32


@dataclass
class ImageSession:
    """Jedna ucitana slika i njen poslednji rezultat obrade."""

    image_id: str
    source: SourceImage
    filename: str
    created_at: float
    mask_override: np.ndarray | None = None
    mask_png: bytes | None = None
    ink_png: bytes | None = None
    last_result: ProcessingResult | None = None
    last_params: dict[str, Any] | None = None
    revision: int = 0


_sessions: dict[str, ImageSession] = {}
_sessions_lock = threading.Lock()

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Sesije i pomocne funkcije
# ---------------------------------------------------------------------------

def _evict_sessions() -> None:
    if len(_sessions) <= MAX_SESSIONS:
        return
    oldest = sorted(_sessions.values(), key=lambda item: item.created_at)
    for session in oldest[:len(_sessions) - MAX_SESSIONS]:
        _sessions.pop(session.image_id, None)


def _get_session(image_id: str) -> ImageSession:
    session = _sessions.get(image_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Slika nije ucitana (istekla sesija)")
    return session


def _store_result(session: ImageSession, result: ProcessingResult,
                  params: ProcessParams) -> None:
    session.last_result = result
    session.last_params = params.to_dict()
    session.ink_png = result.ink_png
    if result.mask_png is not None and session.mask_override is None:
        session.mask_png = result.mask_png
    session.revision += 1


def _svg_response(content: str, *, filename: str | None = None) -> Response:
    headers = {"Cache-Control": "no-store"}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(content=content, media_type="image/svg+xml", headers=headers)


def _render_meta(session: ImageSession, result: ProcessingResult) -> dict[str, Any]:
    base = f"/api/images/{session.image_id}"
    payload = {
        "document_id": f"{session.image_id}:{session.revision}",
        "svg_url": f"{base}/svg?v={session.revision}",
        "width_mm": round(result.width_mm, 3),
        "height_mm": round(result.height_mm, 3),
        "stats": result.stats.as_dict(),
        "timings_ms": {key: round(value, 1) for key, value in result.timings_ms.items()},
        "warnings": list(result.warnings),
        "coverage": result.coverage,
        "mode": result.mode,
        "mask_url": f"{base}/mask?v={session.revision}" if session.mask_png else None,
        "ink_url": f"{base}/ink?v={session.revision}" if session.ink_png else None,
    }
    return payload


def _render_with(session: ImageSession, params: ProcessParams) -> tuple[ProcessingResult, dict[str, Any]]:
    if (session.last_result is not None and session.last_params == params.to_dict()):
        return session.last_result, _render_meta(session, session.last_result)
    result = process_image(session.source, params, mask_override=session.mask_override)
    _store_result(session, result, params)
    return result, _render_meta(session, result)


# ---------------------------------------------------------------------------
# Faza 3: asinhroni red cekanja
# ---------------------------------------------------------------------------

def _update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _run_job(job_id: str, image_id: str, params: ProcessParams) -> None:
    _update_job(job_id, status="running", progress=10, message="Priprema modela dubine...")
    try:
        session = _sessions.get(image_id)
        if session is None:
            raise RuntimeError("Sesija slike je istekla")
        _update_job(job_id, progress=35, message="Estimacija dubine i HLR...")
        result = process_image(session.source, params,
                               mask_override=session.mask_override)
        _update_job(job_id, progress=90, message="Izvoz vektora...")
        _store_result(session, result, params)
        with _jobs_lock:
            if len(_jobs) > MAX_JOBS_KEPT:
                for old_id in list(_jobs)[:len(_jobs) - MAX_JOBS_KEPT]:
                    _jobs.pop(old_id, None)
        _update_job(job_id, status="done", progress=100,
                    message="Gotovo",
                    result=_render_meta(session, result))
    except Exception as exc:  # noqa: BLE001 - greska se prijavljuje UI-ju
        _update_job(job_id, status="error", progress=100, message=str(exc))


# ---------------------------------------------------------------------------
# Aplikacija
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # rembg se predgreva u pozadini: nikada tokom korisnickog zahteva
    def _warmup() -> None:
        warmup("isnet-general-use")
    threading.Thread(target=_warmup, name="rembg-warmup", daemon=True).start()
    yield


app = FastAPI(title="img2vec-lightburn", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> Response:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():  # pragma: no cover
        raise HTTPException(status_code=500, detail="UI nije instaliran")
    return Response(content=index_path.read_text(encoding="utf-8"),
                    media_type="text/html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from img2vec.phase3 import depth_model_path
    return HealthResponse(
        ok=True,
        version=__version__,
        potrace=find_potrace() is not None,
        rembg_ready=is_session_ready(),
        depth_ready=depth_model_path().is_file(),
        providers=available_providers(),
        modes=MODE_LABELS,
        max_upload_mb=round(MAX_INPUT_BYTES / 1024 / 1024, 1),
    )


@app.post("/api/images")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read(MAX_INPUT_BYTES + 1)
    if len(content) > MAX_INPUT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fajl je prevelik (maksimum {MAX_INPUT_BYTES // 1024 // 1024} MB)")
    try:
        source = load_image_bytes(content, filename=file.filename or "slika")
    except ImageLoadError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    image_id = uuid.uuid4().hex[:16]
    with _sessions_lock:
        _sessions[image_id] = ImageSession(
            image_id=image_id,
            source=source,
            filename=file.filename or "slika",
            created_at=time.time(),
        )
        _evict_sessions()
    return JSONResponse({
        "image_id": image_id,
        "filename": file.filename,
        "width": source.width,
        "height": source.height,
        "original_size": list(source.original_size),
        "has_alpha": source.alpha is not None,
        "downscaled": source.work_size != source.original_size,
        "original_url": f"/api/images/{image_id}/original",
    })


@app.get("/api/images/{image_id}/original")
def image_original(image_id: str) -> Response:
    session = _get_session(image_id)
    png = encode_png(session.source.rgb)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/images/{image_id}/mask")
def image_mask(image_id: str) -> Response:
    session = _get_session(image_id)
    png = session.mask_png
    if png is None and session.mask_override is not None:
        png = encode_png(session.mask_override)
    if png is None:
        raise HTTPException(status_code=404, detail="Maska jos nije izracunata")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/images/{image_id}/ink")
def image_ink(image_id: str) -> Response:
    session = _get_session(image_id)
    if session.ink_png is None:
        raise HTTPException(status_code=404, detail="Binarna gravura nije dostupna")
    return Response(content=session.ink_png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/images/{image_id}/mask")
def set_manual_mask(image_id: str, request: MaskRequest) -> JSONResponse:
    """Postavlja rucno ociscenu masku (cetkica/gumica) iz UI canvasa."""
    session = _get_session(image_id)
    raw = request.mask_png
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Neispravna maska (PNG base64)") from exc
    if array is None:
        raise HTTPException(status_code=400, detail="Maska se ne moze dekodirati")
    if array.shape != (session.source.height, session.source.width):
        array = cv2.resize(array, (session.source.width, session.source.height),
                           interpolation=cv2.INTER_NEAREST)
    session.mask_override = ((array > 127).astype(np.uint8) * 255)
    session.mask_png = encode_png(session.mask_override)
    session.last_result = None
    return JSONResponse({"ok": True, "mask_url": f"/api/images/{image_id}/mask"})


@app.delete("/api/images/{image_id}/mask")
def clear_manual_mask(image_id: str) -> JSONResponse:
    session = _get_session(image_id)
    session.mask_override = None
    session.last_result = None
    return JSONResponse({"ok": True})


@app.post("/api/render", response_model=RenderResponse)
def render(request: RenderRequest) -> RenderResponse:
    session = _get_session(request.image_id)
    try:
        params = ProcessParams.from_dict(request.params)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Neispravni parametri: {exc}") from exc
    if params.mode == "wireframe":
        raise HTTPException(
            status_code=422,
            detail="Faza 3 se pokrece asinhrono preko /api/jobs")
    try:
        _result, meta = _render_with(session, params)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RenderResponse(**meta)


@app.get("/api/images/{image_id}/svg")
def get_svg(image_id: str) -> Response:
    session = _get_session(image_id)
    if session.last_result is None:
        raise HTTPException(status_code=404, detail="Nema rezultata obrade")
    return _svg_response(session.last_result.svg)


@app.get("/api/images/{image_id}/download/svg")
def download_svg(image_id: str) -> Response:
    session = _get_session(image_id)
    if session.last_result is None:
        raise HTTPException(status_code=404, detail="Nema rezultata obrade")
    stem = Path(session.filename).stem or "vektor"
    return _svg_response(session.last_result.svg, filename=f"{stem}.svg")


@app.get("/api/images/{image_id}/download/dxf")
def download_dxf(image_id: str) -> Response:
    session = _get_session(image_id)
    if session.last_result is None or not session.last_result.dxf:
        raise HTTPException(status_code=404, detail="Nema DXF rezultata")
    stem = Path(session.filename).stem or "vektor"
    return Response(
        content=session.last_result.dxf,
        media_type="application/dxf",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.dxf"',
            "Cache-Control": "no-store",
        })


@app.get("/api/images/{image_id}/preview.png")
def render_preview_png(image_id: str) -> Response:
    session = _get_session(image_id)
    if session.last_result is None:
        raise HTTPException(status_code=404, detail="Nema rezultata obrade")
    result = session.last_result
    png = encode_preview_png(result.paths, result.width_mm, result.height_mm,
                            max_px=1600)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Faza 3 job queue
# ---------------------------------------------------------------------------

@app.post("/api/jobs", response_model=JobResponse)
def create_job(request: JobRequest) -> JobResponse:
    _get_session(request.image_id)
    try:
        params = ProcessParams.from_dict(request.params)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Neispravni parametri: {exc}") from exc
    if params.mode != "wireframe":
        params.mode = "wireframe"
    try:
        params.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": 0,
                         "message": "U redu cekanja", "result": None}
    threading.Thread(target=_run_job, args=(job_id, request.image_id, params),
                     name=f"job-{job_id}", daemon=True).start()
    return JobResponse(job_id=job_id, status="queued", progress=0,
                       message="U redu cekanja")


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Nepoznat posao")
    return JobResponse(**job)


# ---------------------------------------------------------------------------
# Test dataset (ugradjeni primeri iz test_images/)
# ---------------------------------------------------------------------------

@app.get("/api/test-images")
def list_test_images() -> JSONResponse:
    from img2vec.config import TEST_IMAGES_DIR

    if not TEST_IMAGES_DIR.is_dir():
        return JSONResponse({"images": []})
    names = sorted(path.name for path in TEST_IMAGES_DIR.glob("*.png"))
    return JSONResponse({"images": names})


@app.get("/api/test-images/{name}")
def get_test_image(name: str) -> Response:
    from img2vec.config import TEST_IMAGES_DIR

    safe_name = Path(name).name
    candidate = TEST_IMAGES_DIR / safe_name
    if candidate.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(status_code=400, detail="Nepodrzan format")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Test slika nije pronadjena")
    return Response(content=candidate.read_bytes(), media_type="image/png")


# ---------------------------------------------------------------------------
# Preseti
# ---------------------------------------------------------------------------

@app.get("/api/presets")
def get_presets() -> JSONResponse:
    return JSONResponse({"presets": [preset.as_dict() for preset in list_presets()]})


@app.post("/api/presets")
def post_preset(request: PresetRequest) -> JSONResponse:
    try:
        params = ProcessParams.from_dict(request.params)
        preset = save_preset(request.key, request.name, request.description, params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "preset": preset.as_dict()})


@app.delete("/api/presets/{key}")
def remove_preset(key: str) -> JSONResponse:
    try:
        removed = delete_preset_file(key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Preset nije pronadjen")
    return JSONResponse({"ok": True})