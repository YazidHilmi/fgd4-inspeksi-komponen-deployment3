"""Membandingkan skor aplikasi dengan skor yang disimpan notebook (prediksi_semua_model.csv).

Contoh:
    python scripts/verify_against_notebook.py \
        --csv models/prediksi_semua_model.csv \
        --dataset "/path/ke/mvtec_anomaly_detection" --per-group 3

``--dataset`` adalah folder yang berisi ``screw/`` dan ``metal_nut/`` (struktur MVTec AD asli:
``<kategori>/test/<jenis>/<file>.png``). Skrip menjalankan InspectionEngine yang sama dengan
aplikasi dan melaporkan selisih skor terhadap CSV. Selisih kecil (kira-kira < 1e-3) menandakan
pipeline aplikasi sama dengan notebook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspection import config as cfg  # noqa: E402
from inspection.engine import InspectionEngine, load_rgb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="prediksi_semua_model.csv dari notebook")
    parser.add_argument("--dataset", required=True, help="folder MVTec AD (berisi screw/ dan metal_nut/)")
    parser.add_argument("--per-group", type=int, default=3, help="jumlah gambar per kategori & model")
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    config = cfg.load_config()
    missing = cfg.check_artifacts(config)
    if missing:
        print("Berkas belum lengkap:\n  - " + "\n  - ".join(missing))
        return 2
    engine = InspectionEngine(config)
    table = pd.read_csv(args.csv)
    worst = 0.0
    failures = 0
    for (category, model), group in table.groupby(["kategori", "model"]):
        if category not in cfg.CATEGORIES:
            continue
        if model != cfg.MODEL_EFFICIENTAD and config.bank_path(category, model) is None:
            print(f"[lewati] {category} | {model}: bank tidak ada di models/")
            continue
        sample = pd.concat([group[group.label_asli == 0].head(max(1, args.per_group // 2)), group[group.label_asli == 1].head(args.per_group)])
        for row in sample.itertuples(index=False):
            path = Path(args.dataset) / category / "test" / row.sample_key
            if not path.is_file():
                print(f"[lewati] berkas tidak ada: {path}")
                continue
            out = engine.run(load_rgb(path), category, model)
            diff = abs(out.image_score - row.image_score)
            same_decision = out.is_defect == bool(row.prediksi)
            worst = max(worst, diff)
            failures += int(diff > args.tol or not same_decision)
            print(f"{category:9s} {model:22s} {row.sample_key:28s} app={out.image_score:.6f} notebook={row.image_score:.6f} "
                  f"|selisih|={diff:.2e} keputusan {'sama' if same_decision else 'BEDA'}")
    print(f"\nSelisih terbesar: {worst:.3e} | pengecekan gagal: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
