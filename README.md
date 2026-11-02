# Image-to-Vector za Laser / LightBurn (v2.0)

Modularni interni sistem (**CLI + Web aplikacija**) koji rastersku sliku
automatski konvertuje u proizvodno spreman vektorski format (**SVG / DXF**) sa
tačnim fizičkim dimenzijama u milimetrima, optimizovan za direktan uvoz u
**LightBurn**.

Poseban fokus: razdvajanje pravih jednolinijskih putanja (*Centerline*) od
površinskih gravura (*Stencil*), automatski **Cut obris** i minimizacija
praznog hoda laserske glave.

---

## ✨ Ključne mogućnosti

| Oblast | Detalj |
| --- | --- |
| **4 režima** | Stencil (Fill), Centerline (Line), 2D low-poly mesh (Faza 2), 3D wireframe + HLR (Faza 3) |
| **Milimetri** | Ceo pipeline radi u mm; SVG `width="150mm"` + `viewBox="0 0 150 192.5"` |
| **LightBurn paleta** | Crna `#000000` (gravura), plava `#0000FF` (detalj), crvena `#FF0000` (Cut) |
| **AI segmentacija** | `rembg` (ISNet / BiRefNet ONNX), CUDA → CoreML → CPU fallback, modeli u `./models` |
| **Automatski prag** | Otsu za bimodalne (silueta/logo/tekst), adaptive za fotografije |
| **Cut obris** | `shapely` buffer offset (0–15 mm), glatki `join_style=round`, simplify 0.05 mm |
| **Optimizacija** | Skeleton + `skan` graf + spur pruning, vpype linemerge, RDP, nearest-neighbour sort |
| **Izvoz** | SVG 1.1 (mm) + DXF R2010 (`$INSUNITS = 4`, lejeri `00_ENGRAVE` / `01_DETAIL` / `02_CUT`) |
| **Web UI** | HTML/JS (Alpine.js), SVG pregled sa zoom/pan, slojevi, preseti, live statistika |
| **Preseti** | Ugrađeni (LED akril, drvo, stencil, line-art) + timski JSON u `presets/` |
| **Dataset** | 10 referentnih slika + verifikacioni spike (acceptance **10/10**) |

---

## 🚀 Brzi start (Linux / macOS)

```bash
# 1) Python 3.12 okruzenje i sve zavisnosti (uv preuzima tacan Python)
uv sync

# 2) potrace (vektorizacija siluete) - lokalni build bez sudo prava
uv run python scripts/setup_potrace.py
#    alternativa: pacman -S potrace / brew install potrace / apt install potrace

# 3) AI modeli (preuzimaju se jednom, nikada tokom korisnickog zahteva)
uv run python scripts/download_models.py        # rembg ISNet (~179 MB)
#    uv run python scripts/download_models.py --all   # + BiRefNet + Depth Anything V2

# 4) Web aplikacija (pali server i otvara browser)
uv run python run_app.py
```

Preduslovi: `uv` (<https://docs.astral.sh/uv/>), `git`; za lokalni potrace build
još `make` i C kompajler (`gcc` / `cc`). Bez sistemskih Python paketa - sve ide u
projektni `.venv`.

> Na macOS-u su portovi 5000/7000 zauzeti (AirPlay), zato podrazumevani port je
> **8765** sa automatskim fallback-om na prvi slobodan port.

---

## 🖥️ Web UI

`uv run python run_app.py` podiže FastAPI server i otvara interfejs:

* **Leva kolona** - drag & drop / izbor slike, test dataset, izbor režima,
  material preseti, AI uklanjanje pozadine.
* **Centralna kolona** - SVG vektor sa *zoom/pan*, tabovi *Vektor / Original /
  Maska / Gravura*, uključivanje/isključivanje sloja gravure i Cut-a.
* **Desna kolona** - dimenzije u mm, Cut offset, parametri obrade, live
  statistika (broj putanja, čvorova, dužina lasera) i izvoz SVG/DXF/PNG.

Deep link za brzu proveru: `http://127.0.0.1:8765/?demo=03_logo_bw.png`

REST API (skraćeno):

| Ruta | Namena |
| --- | --- |
| `POST /api/images` | upload (limit **10 MB**, max 2000 px duža strana) |
| `POST /api/render` | obrada + metadata + URL do SVG-a |
| `GET /api/images/{id}/svg` | inline SVG |
| `GET /api/images/{id}/download/svg` / `dxf` | preuzimanje |
| `POST /api/images/{id}/mask` | ručna maska (četkica/gumica) |
| `GET/POST/DELETE /api/presets` | timski preseti |
| `POST /api/jobs` + `GET /api/jobs/{id}` | asinhrona Faza 3 sa progresom |
| `GET /api/health` | status potrace / rembg / depth modela |

---

## ⌨️ CLI

```bash
# Osnovna upotreba (Stencil + crveni Cut offset 3 mm)
uv run python -m img2vec slika.png -o out --width 150 --cut-offset 3

# Centerline gravura sa presetom
uv run python -m img2vec slika.png --mode centerline --preset led-akril -o out

# Batch obrada celog direktorijuma, SVG + DXF + preview
uv run python -m img2vec test_images/ -o out/batch --preview

# Masinski citljiv izlaz za skripte
uv run python -m img2vec slika.png --json

# Lista dostupnih preseta
uv run python -m img2vec --list-presets
```

Najvažnije opcije: `--mode {stencil,centerline,mesh,wireframe}`, `--width`,
`--height`, `--cut-offset`, `--no-cut`, `--no-rembg`, `--preset`, `--formats`,
`--preview`, `--json`. Konzolna komanda je dostupna i kao `uv run img2vec ...`
(editable install).

---

## 📏 LightBurn standardizacija

1. **Koordinatni sistem:** 1 SVG jedinica = 1 mm. Root element uvek ima
   `width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}"`, bez 72/96 DPI zabune.
2. **Slojevi (boje):**

   | Sloj | Boja | Namena |
   | --- | --- | --- |
   | `00_ENGRAVE` | `#000000` | glavna gravura - fill poligoni (Stencil) ili linije (Centerline) |
   | `01_DETAIL` | `#0000FF` | pomoćni detalji / score |
   | `02_CUT` | `#FF0000` | spoljni obris za sečenje, uvek `fill="none"`, `stroke-width="0.1"` |

3. **Cut offset** proširuje platno za `2 × offset` mm, tako da crvena linija
   uvek ostane unutar dokumenta: `150 mm + 2 × 3 mm = 156 mm`.
4. **DXF** je R2010 sa `$INSUNITS = 4` (mm), razdvojenim imenovanim lejerima i
   Bezier krivama flatten-ovanim na 0.05 mm.
5. **Zaštita:** max ulaz **10 MB**, automatski downscale na **2000 px** duže
   strane (rezolucija lasera 0.1 mm i dalje ostaje sačuvana).

---

## 🧪 Test dataset i acceptance

`test_images/` sadrži 10 deterministički generisanih referentnih slika
(portret, silueta, logo, tekst sa serifima, line-art, kompleksna pozadina,
sitni detalji, niska/visoka rezolucija, transparentni PNG). Detalji:
`test_images/README.md`, metapodaci: `test_images/manifest.json`.

```bash
# regenerisanje dataseta (offline, bez autorskih prava)
uv run python scripts/generate_test_dataset.py

# verifikacioni spike Faze 1 (sva 3 puta: stencil, centerline, cut)
uv run python spike_phase1.py
```

Rezultat na ovoj mašini (sa rembg maskom): **10/10 slika daje čist SVG** za
preporučeni režim (kriterijum Faze 1: ≥ 8/10), prosečno ~2.7 s po slici.
Izveštaj: `out/spike/spike_report.json`.

---

## ✅ Testovi

```bash
uv run python -m pytest              # jedinicni + API testovi
uv run python -m pytest -m slow      # acceptance nad celim datasetom
```

Pokriveno: mm dimenzije i viewBox, LightBurn paleta, compound rupe
(`fill-rule=evenodd`), centerline daje **jednu** putanju (ne duplu konturu),
Cut offset geometrija, DXF lejeri i `$INSUNITS = 4`, limit veličine fajla,
alfa kanal, downscale, API rute i preseti.

---

## 🗂️ Struktura projekta

```text
.
├── backend/                  # FastAPI aplikacija + Web UI (static/)
│   ├── main.py               # rute, sesije, preseti, Faza 3 job queue
│   ├── schemas.py            # Pydantic modeli
│   └── static/               # index.html, app.js, style.css, alpine.min.js
├── src/img2vec/              # produkcijsko jezgro
│   ├── config.py             # ProcessParams (JSON preset) + limiti
│   ├── image_io.py           # ucitavanje, 10 MB limit, alfa, downscale
│   ├── segment.py            # rembg ONNX sesija (pre-warm), alfa, ciscenje maske
│   ├── stencil.py            # auto threshold -> potrace (Fill)
│   ├── centerline.py         # XDoG/Otsu -> skeleton -> skan -> spur pruning
│   ├── potrace.py            # adapter za sistemski CLI + SVG parser
│   ├── cutout.py             # shapely buffer offset (Cut sloj)
│   ├── optimize.py           # linemerge, RDP, nearest-neighbour sort
│   ├── geometry.py           # model putanja u mm, SVG path parser
│   ├── svg_export.py         # LightBurn SVG (mm, evenodd compound)
│   ├── dxf_export.py         # DXF R2010 sa lejerima
│   ├── pipeline.py           # orkestracija Faze 1 (+ mask override)
│   ├── phase2.py             # 2D low-poly mesh (Delaunay + dedup ivica)
│   ├── phase3.py             # 3D wireframe (Depth Anything V2 + HLR)
│   ├── preview.py            # raster preview (Web UI / debug)
│   ├── presets.py            # ugradjeni + timski JSON preseti
│   └── cli.py                # komandna linija
├── scripts/                  # generator dataseta, setup potrace, modeli
├── tests/                    # pytest jedinicni, API i acceptance testovi
├── test_images/              # 10 referentnih slika + manifest
├── presets/                  # timski JSON preseti (u git-u)
├── models/                   # ONNX kes (gitignore; preuzima se automatski)
├── spike_phase1.py           # verifikacioni spike Faze 1
├── run_app.py / dev.py       # lokalni launcheri (bez bash skripti)
└── pyproject.toml            # uv zavisnosti + uv.lock
```

---

## 🔬 Režimi obrade

### Stencil / Silhouette (LightBurn *Fill*)
`grayscale → bilateral → auto prag (Otsu/adaptive) → morfologija → min_island →
potrace --flat (glatke Bezier krive) → mm → SVG evenodd compound`

### Centerline (LightBurn *Line*)
`grayscale → XDoG ili auto prag → skeletonize → skan graf → uklanjanje slepih
grana (min_spur_px) → mm → vpype linemerge → RDP 0.05 mm → linesort`

### Cut obris
`maska subjekta → cv2.findContours → shapely Polygon → buffer(offset,
join_style=round) → simplify(0.05) → crveni sloj #FF0000`

### Faza 2 - 2D low-poly mesh
`Delaunay nad tackama (gustina po gradijentu ivica) → brisanje trouglova van
maske → deduplikacija ivica set(sorted(pair)) → linemerge → mm`

### Faza 3 - 3D wireframe iluzija
`Depth Anything V2 ONNX → bilateralno glacanje → izolinije dubine + vertikalna
mreza → pomeranje po Z (paralaksa) → z-buffer Hidden Line Removal → mm`

> Faza 3 se u Web UI-ju pokreće asinhrono (red čekanja + progress bar), jer
> CPU inferencija traje nekoliko sekundi.

---

## 🎛️ Preseti

Ugrađeni preseti (u kodu): `led-akril`, `drvo-portret`, `stencil-kontura`,
`line-art`. Timski preseti se čuvaju kao JSON u `presets/` i dele kroz git
(Web UI: *Sačuvaj trenutne parametre kao preset*; CLI: `--preset <kljuc>`).

```json
{
  "name": "LED Akrilna Lampa",
  "description": "Centerline gravura + Cut offset 3 mm",
  "params": { "mode": "centerline", "add_cut": true, "cut_offset_mm": 3.0 }
}
```

---

## 🌍 Cross-platform napomene

* Fiksiran Python **3.12** (`.python-version` + `uv.lock`) - isti lock radi na
  Linuxu, macOS-u i Windowsu.
* `onnxruntime` (osnovna zavisnost) bira izvršni provajder dinamički:
  `CUDAExecutionProvider` → `CoreMLExecutionProvider` → `CPUExecutionProvider`.
* Nula sistemskih Python zavisnosti: `opencv-python-headless`, `shapely`,
  `scikit-image`, `vpype`, `skan`, `ezdxf` imaju gotove binary wheel-ove.
* Potrace je spoljni CLI (po specifikaciji): `setup_potrace.py` ga kompajlira
  lokalno u `vendor/` bez sudo prava; alternativno sistemski paket.
* Putanje u kodu isključivo preko `pathlib.Path` - nema ručnog spajanja /.
* `.gitattributes` forsira LF završetke; `.gitignore` isključuje modele i
  `.venv` (aplikacija sama preuzima ONNX modele pri prvom pokretanju).

---

## 🧭 Ograničenja i sledeći koraci (van Faze 1)

* Svesno odloženo: batch ZIP obrada kroz UI, nativni `.lbrn2` XML eksport,
  domaći (stopica/tab) cut dodaci, engleska lokalizacija UI-ja.
* Faza 2 i Faza 3 su integrisane, ali su u ranoj fazi - parametri
  (`mesh_spacing_mm`, `depth_levels`, `depth_parallax`) su izloženi u UI-ju.
* `vpype` 1.15 je refaktorisao bibliotečni API (nema `linesimplify/linesort`),
  pa su RDP i nearest-neighbour sort implementirani u `optimize.py`, a vpype
  `LineCollection.merge()` se koristi za linemerge.

---

## 📄 Licenca

Interni projekat - sva prava zadržana.
