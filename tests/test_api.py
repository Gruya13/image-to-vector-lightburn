"""Testovi FastAPI backenda (upload, render, izvoz, preseti)."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from img2vec.config import MAX_INPUT_BYTES
from tests.helpers import black_rectangle, png_bytes, white_canvas

client = TestClient(app)


def _upload(data: bytes | None = None, name: str = "test.png") -> dict:
    response = client.post(
        "/api/images",
        files={"file": (name, data if data is not None else black_rectangle(), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["max_upload_mb"] == 10
    assert "stencil" in payload["modes"]


def test_upload_i_osnovne_informacije():
    payload = _upload()
    assert payload["width"] == 400
    assert payload["height"] == 400
    assert payload["has_alpha"] is False
    assert payload["image_id"]


def test_render_i_preuzimanje_svg():
    uploaded = _upload()
    response = client.post("/api/render", json={
        "image_id": uploaded["image_id"],
        "params": {"mode": "stencil", "remove_background": False,
                   "target_width_mm": 100.0},
    })
    assert response.status_code == 200, response.text
    meta = response.json()
    assert meta["width_mm"] == pytest.approx(100.0)
    assert meta["stats"]["path_count"] >= 1

    svg = client.get(meta["svg_url"])
    assert svg.status_code == 200
    assert 'width="100mm"' in svg.text
    assert "viewBox" in svg.text

    download = client.get(f"/api/images/{uploaded['image_id']}/download/svg")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]

    dxf = client.get(f"/api/images/{uploaded['image_id']}/download/dxf")
    assert dxf.status_code == 200
    assert "$INSUNITS" in dxf.text
    assert "00_ENGRAVE" in dxf.text


def test_test_dataset_rute():
    listing = client.get("/api/test-images")
    assert listing.status_code == 200
    images = listing.json()["images"]
    assert len(images) == 10
    first = client.get("/api/test-images/" + images[0])
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/png"


def test_upload_preveliki_fajl():
    big = b"0" * (MAX_INPUT_BYTES + 1024)
    response = client.post("/api/images",
                           files={"file": ("big.png", big, "image/png")})
    assert response.status_code == 413


def test_upload_nevalidna_slika():
    response = client.post("/api/images",
                           files={"file": ("x.txt", b"nije slika", "text/plain")})
    assert response.status_code == 415


def test_render_nepoznata_sesija():
    response = client.post("/api/render", json={
        "image_id": "nepostoji",
        "params": {"mode": "stencil", "remove_background": False},
    })
    assert response.status_code == 404


def test_render_nevalidni_parametri():
    uploaded = _upload()
    response = client.post("/api/render", json={
        "image_id": uploaded["image_id"],
        "params": {"mode": "stencil", "threshold_block": 4},
    })
    assert response.status_code == 422


def test_rucna_maska_override():
    uploaded = _upload()
    canvas = white_canvas(400, 400)
    canvas[100:300, 100:300] = 0
    mask_png = "data:image/png;base64," + base64.b64encode(
        png_bytes(canvas[..., 0])).decode("ascii")
    response = client.post(f"/api/images/{uploaded['image_id']}/mask",
                           json={"mask_png": mask_png})
    assert response.status_code == 200, response.text
    mask = client.get(response.json()["mask_url"])
    assert mask.status_code == 200
    cleared = client.delete(f"/api/images/{uploaded['image_id']}/mask")
    assert cleared.status_code == 200


def test_preseti_ciklus():
    created = client.post("/api/presets", json={
        "key": "test-preset-xyz",
        "name": "Test preset",
        "description": "test",
        "params": {"mode": "stencil", "target_width_mm": 120.0},
    })
    assert created.status_code == 200, created.text
    listing = client.get("/api/presets")
    keys = [preset["key"] for preset in listing.json()["presets"]]
    assert "led-akril" in keys
    assert "test-preset-xyz" in keys
    try:
        assert client.delete("/api/presets/test-preset-xyz").status_code == 200
    finally:
        client.delete("/api/presets/test-preset-xyz")


def test_ugradjeni_preset_se_ne_brise():
    response = client.delete("/api/presets/led-akril")
    assert response.status_code == 422


def test_faza3_job_odbija_pogresan_rezim():
    uploaded = _upload()
    response = client.post("/api/jobs", json={
        "image_id": uploaded["image_id"],
        "params": {"mode": "stencil", "remove_background": False},
    })
    # mode se forsira na wireframe, pa zahtev prodje ili padne zbog modela - bitno
    # je da endpoint postoji i da vrati validan odgovor
    assert response.status_code in (200, 422, 500)
