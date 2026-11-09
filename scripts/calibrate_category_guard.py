"""Mengkalibrasi batas penjaga "bukan sekrup/mur" memakai gambar nyata (opsional, disarankan).

Tanpa berkas ini, aplikasi memakai batas otomatis dari bank referensi 16-shot. Skrip ini membuat
``models/category_guard.json`` yang lebih akurat dari:
  * gambar test MVTec asli (normal + semua jenis cacat) tiap kategori -> batas HARUS di atas nilai tertinggi;
  * opsional ``--negatives``: folder berisi foto BUKAN sekrup/mur (benda lain, orang, ruangan, dll.)
    -> batas diletakkan di antara nilai tertinggi gambar komponen dan nilai terendah gambar negatif.

Contoh:
    python scripts/calibrate_category_guard.py --dataset "/path/ke/mvtec_anomaly_detection" --negatives "/path/ke/foto_lain"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection import config as cfg  # noqa: E402
from inspection.engine import InspectionEngine, load_rgb  # noqa: E402

EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="folder MVTec AD (berisi screw/ dan metal_nut/)")
    parser.add_argument("--negatives", help="folder foto yang BUKAN sekrup/mur (opsional)")
    parser.add_argument("--margin", type=float, default=1.3, help="pengali nilai tertinggi komponen bila tidak ada negatif")
    args = parser.parse_args()

    config = cfg.load_config()
    missing = cfg.check_artifacts(config)
    if missing:
        print("Berkas belum lengkap:\n  - " + "\n  - ".join(missing))
        return 2
    engine = InspectionEngine(config)
    engine.preload(cfg.CATEGORIES[0], cfg.MODEL_EFFICIENTAD)

    inside = {c: [] for c in cfg.CATEGORIES}   # P90 ke bank kategorinya sendiri, gambar test kategori itu
    cross = {c: [] for c in cfg.CATEGORIES}    # P90 ke bank kategori c, gambar kategori LAIN
    for category in cfg.CATEGORIES:
        files = images_in(Path(args.dataset) / category / "test")
        print(f"{category}: {len(files)} gambar test")
        for path in files:
            _, tail = engine.category_stats(load_rgb(path))
            for c in cfg.CATEGORIES:
                (inside if c == category else cross)[c].append(tail[c])

    negatives = {c: [] for c in cfg.CATEGORIES}
    if args.negatives:
        files = images_in(Path(args.negatives))
        print(f"negatif: {len(files)} gambar")
        for path in files:
            _, tail = engine.category_stats(load_rgb(path))
            for c in cfg.CATEGORIES:
                negatives[c].append(tail[c])

    guards, report = {}, {}
    for c in cfg.CATEGORIES:
        top = max(inside[c])
        low_cross = min(cross[c]) if cross[c] else float("nan")
        low_neg = min(negatives[c]) if negatives[c] else None
        if low_neg is not None:
            guard = math.sqrt(top * low_neg) if low_neg > top else top * args.margin
            note = "OK: ada celah antara komponen dan negatif" if low_neg > top else "PERINGATAN: negatif ada yang lebih dekat dari komponen; periksa foto negatif"
        else:
            guard, note = top * args.margin, "tanpa negatif: batas = tertinggi x margin"
        guards[c] = guard
        report[c] = {"komponen_maks": top, "kategori_lain_min": low_cross, "negatif_min": low_neg, "batas": guard, "catatan": note}
        print(f"[{c}] komponen maks={top:.4f} | kategori lain min={low_cross:.4f} | negatif min={low_neg} | batas={guard:.4f} -> {note}")

    out = cfg.MODELS_DIR / cfg.GUARD_FILENAME
    out.write_text(json.dumps({"statistic": f"p{cfg.OOD_PERCENTILE:g}", "bank": cfg.CATEGORY_CHECK_BANK, "guards": guards, "report": report}, indent=2), encoding="utf-8")
    print("Tersimpan:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
