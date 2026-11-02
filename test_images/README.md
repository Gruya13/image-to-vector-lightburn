# Referentni test dataset (Faza 1)

10 deterministički generisanih slika koje pokrivaju sve kritične slučajeve iz
plana. Regenerisanje je potpuno offline i bez autorskih prava:

```bash
uv run python scripts/generate_test_dataset.py
```

| # | Fajl | Kategorija | Šta testira |
| --- | --- | --- | --- |
| 1 | `01_portret_foto.png` | Portret (foto) | kvalitet rembg maske, XDoG/adaptive detalji |
| 2 | `02_silueta_figura.png` | Silueta | Stencil mod + tačnost Cut offseta |
| 3 | `03_logo_bw.png` | Logo (rasterski) | oštrina uglova, compound rupe |
| 4 | `04_tekst_serif.png` | Tekst / tipografija | očuvanje unutrašnjih šupljina slova |
| 5 | `05_crtez_line_art.png` | Crtež (line art) | skeletonizacija i spur pruning |
| 6 | `06_kompleksna_pozadina.png` | Kompleksna pozadina | otpornost rembg segmentacije |
| 7 | `07_sitni_detalji.png` | Sitni detalji | filter malih ostrva |
| 8 | `08_niska_rezolucija_300.png` | Niska rezolucija | upscale / bilateralno glačanje |
| 9 | `09_visoka_rezolucija_4k.png` | Visoka rezolucija | automatski downscale i memorija |
| 10 | `10_transparentni_stiker.png` | Transparentni PNG | direktno čitanje alfa kanala |

Kategorije i dimenzije su u `manifest.json` (koristi ih `spike_phase1.py` i
acceptance test). **Acceptance kriterijum Faze 1: >= 8/10 čistih SVG-ova.**
