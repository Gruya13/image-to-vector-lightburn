"""Regresioni testovi za Web UI (sprecava 'mrtav UI' bug)."""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "backend" / "static"


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_app_js_se_ucitava_pre_alpinea():
    """Alpine 3 CDN poziva start() preko queueMicrotask; app() mora biti definisan.

    Ako se alpine.min.js ucita pre app.js, x-data="app()" padne i ceo UI ostane
    bez ijednog event handlera (file picker se otvori, ali se nista ne desi).
    """
    html = _html()
    app_tag = html.index('<script defer src="/static/app.js')
    alpine_tag = html.index('<script defer src="/static/vendor/alpine.min.js"')
    assert app_tag < alpine_tag, "app.js mora biti PRE alpine.min.js"


def _static_path(url_path: str) -> Path:
    return STATIC / url_path.split("/static/", 1)[1]


def test_svi_referencirani_static_fajlovi_postoje():
    html = _html()
    for match in re.finditer(r'src="(/static/[^"?]+)', html):
        assert _static_path(match.group(1)).is_file(), (
            f"UI referencira nepostojeci fajl: {match.group(1)}")
    for match in re.finditer(r'href="(/static/[^"?]+)', html):
        assert _static_path(match.group(1)).is_file(), (
            f"UI referencira nepostojeci fajl: {match.group(1)}")


def test_app_js_definise_alpine_komponentu():
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function app()" in source
    for method in ("init()", "uploadFile(file)", "renderNow()", "applyLayers()"):
        assert method in source, f"app.js nema {method}"


def test_index_ima_kes_busting_za_ui_fajlove():
    html = _html()
    assert re.search(r'app\.js\?v=\d+', html), "app.js nema keš-busting verziju"
    assert re.search(r'style\.css\?v=\d+', html), "style.css nema keš-busting verziju"
