"""Konfigurasi aplikasi dan pembacaan ``deployment_config.json`` dari penelitian.

Seluruh nilai penelitian (threshold, pengaturan masking, ukuran input, dsb.) dibaca
dari ``deployment_config.json`` yang dihasilkan notebook (Tahap 11). Tidak ada nilai
threshold yang ditulis manual di kode aplikasi.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT / "vendor"
MODELS_DIR = Path(os.environ.get("INSPEKSI_MODELS_DIR", ROOT / "models")).expanduser().resolve()
CONFIG_FILENAME = "deployment_config.json"

# Urutan sesuai permintaan: metal_nut dan screw.
CATEGORIES = ("metal_nut", "screw")
CATEGORY_LABELS = {"metal_nut": "Metal Nut (mur)", "screw": "Screw (sekrup)"}

MODEL_EFFICIENTAD = "EfficientAD"
MODEL_ANOMALYDINO = "AnomalyDINO"
# Nama varian persis seperti di notebook / deployment_config.json.
DINO_VARIANTS = ("AnomalyDINO full-shot", "AnomalyDINO 16-shot", "AnomalyDINO 1-shot")
DINO_VARIANT_LABELS = {
    "AnomalyDINO full-shot": "Full-shot (seluruh referensi normal)",
    "AnomalyDINO 16-shot": "16-shot",
    "AnomalyDINO 1-shot": "1-shot",
}

# Bank yang dipakai untuk validasi kategori visual (lebih ringan daripada full-shot).
CATEGORY_CHECK_BANK = "AnomalyDINO 16-shot"

# Penjaga "bukan sekrup/mur": statistik = persentil ke-90 jarak patch ke bank referensi normal.
# Tanpa berkas kalibrasi, batas dihitung otomatis dari bank 16-shot (leave-one-image-out) x OOD_FACTOR.
OOD_PERCENTILE = 90.0
OOD_FACTOR = 2.0
OOD_AUTO_MAX_REFERENCES = 8
GUARD_FILENAME = "category_guard.json"   # dihasilkan scripts/calibrate_category_guard.py (opsional)

# Pengaturan visualisasi (sama dengan notebook, sel 43): skala warna heatmap tetap
# 0 .. HEATMAP_MAX_RATIO x pixel threshold; overlay merah alpha 0,55.
HEATMAP_MAX_RATIO = 2.0
OVERLAY_ALPHA = 0.55


class SetupError(RuntimeError):
    """Berkas model/konfigurasi belum lengkap. ``problems`` berisi daftar masalah."""

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.problems = problems or []


@lru_cache(maxsize=None)
def _files_by_name(models_dir: str) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    base = Path(models_dir)
    if base.is_dir():
        for path in base.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path)
    return index


def find_artifact(declared: str | None, category: str | None = None) -> Path | None:
    """Mencari berkas hasil notebook di ``models/``.

    ``deployment_config.json`` menyimpan path Colab/Drive (mis.
    ``/content/drive/MyDrive/.../screw/anomalydino/bank_full-shot.faiss``). Di aplikasi,
    path itu tidak ada, sehingga dicari berdasarkan nama berkas di ``models/``
    (diutamakan yang berada di folder kategori yang sama).
    """
    if not declared:
        return None
    name = PureWindowsPath(declared).name if "\\" in declared else PurePosixPath(declared).name
    candidates = _files_by_name(str(MODELS_DIR)).get(name, [])
    if category:
        in_category = [p for p in candidates if category in p.parts]
        if in_category:
            return sorted(in_category, key=lambda p: len(p.parts))[0]
        # Berkas tanpa folder kategori (mis. bobot DINOv2 bersama) hanya boleh dipakai
        # bila tidak ada folder kategori lain pada path-nya.
        others = [c for c in CATEGORIES if c != category]
        neutral = [p for p in candidates if not any(o in p.parts for o in others)]
        return sorted(neutral, key=lambda p: len(p.parts))[0] if neutral else None
    return sorted(candidates, key=lambda p: len(p.parts))[0] if candidates else None


@dataclass(frozen=True)
class DeploymentConfig:
    raw: dict

    @property
    def map_size(self) -> int:
        return int(self.raw["map_size"])

    @property
    def dino_size(self) -> int:
        return int(self.raw["dino_input_size"])

    @property
    def dino_backbone(self) -> str:
        return str(self.raw["dino_backbone"])

    def _category(self, category: str) -> dict:
        try:
            return self.raw["categories"][category]
        except KeyError as exc:
            raise SetupError(f"Kategori '{category}' tidak ada di {CONFIG_FILENAME}.") from exc

    def masking(self, category: str) -> bool:
        return bool(self._category(category)["masking"])

    def _model(self, category: str, model_name: str) -> dict:
        try:
            return self._category(category)["models"][model_name]
        except KeyError as exc:
            raise SetupError(f"Model '{model_name}' untuk '{category}' tidak ada di {CONFIG_FILENAME}.") from exc

    def thresholds(self, category: str, model_name: str) -> tuple[float, float]:
        """(image_threshold, pixel_threshold) hasil kalibrasi validasi normal."""
        values = self._model(category, model_name)["threshold"]
        return float(values["image"]), float(values["pixel"])

    def bank_path(self, category: str, model_name: str) -> Path | None:
        return find_artifact(self._model(category, model_name).get("bank"), category)

    def efficientad_checkpoint(self, category: str) -> Path | None:
        return find_artifact(self._category(category).get("efficientad_checkpoint"), category)

    def backbone_weights(self) -> Path | None:
        return find_artifact(self.raw.get("backbone_weights"), None)

    def reference_count(self, category: str, model_name: str) -> int | None:
        refs = self._model(category, model_name).get("references")
        return None if refs is None else len(refs)


def load_config() -> DeploymentConfig:
    path = MODELS_DIR / CONFIG_FILENAME
    if not path.is_file():
        found = _files_by_name(str(MODELS_DIR)).get(CONFIG_FILENAME)
        if found:
            path = found[0]
        else:
            raise SetupError(
                f"{CONFIG_FILENAME} tidak ditemukan.",
                [f"Letakkan {CONFIG_FILENAME} di folder: {MODELS_DIR}"],
            )
    raw = json.loads(path.read_text(encoding="utf-8"))
    needed = ["map_size", "dino_backbone", "dino_input_size", "dino_preprocess", "categories"]
    absent = [key for key in needed if key not in raw]
    if absent:
        raise SetupError(f"{CONFIG_FILENAME} tidak lengkap.", [f"Kunci hilang: {', '.join(absent)}"])
    if raw["dino_preprocess"] != "masking_only":
        raise SetupError(
            f"{CONFIG_FILENAME} memakai dino_preprocess='{raw['dino_preprocess']}', sedangkan aplikasi "
            "mengikuti notebook (masking_only).",
        )
    for category in CATEGORIES:
        entry = raw["categories"].get(category)
        if entry is None:
            raise SetupError(f"Kategori '{category}' tidak ada di {CONFIG_FILENAME}.")
        if entry.get("rotation"):
            raise SetupError(
                f"{CONFIG_FILENAME}: kategori '{category}' memakai rotasi pada bank referensi; "
                "bank sudah dibangun, tetapi pengaturan ini tidak sesuai notebook yang dipelajari."
            )
    return DeploymentConfig(raw)


def check_artifacts(config: DeploymentConfig) -> list[str]:
    """Berkas WAJIB yang masih hilang (kosong bila lengkap).

    Wajib: bobot DINOv2 (juga dipakai validasi kategori), checkpoint EfficientAD dan bank
    16-shot (dipakai validasi kategori) untuk kedua kategori. Bank 1-shot/full-shot bersifat
    opsional; varian yang bank-nya tidak ada akan dinonaktifkan di antarmuka.
    """
    problems: list[str] = []
    if config.backbone_weights() is None:
        problems.append("Bobot DINOv2: dinov2_vits14_state_dict.pt")
    for category in CATEGORIES:
        if config.efficientad_checkpoint(category) is None:
            problems.append(f"Checkpoint EfficientAD '{category}': {category}/efficientad/efficientad_final_60epoch.ckpt")
        if config.bank_path(category, CATEGORY_CHECK_BANK) is None:
            problems.append(f"Memory bank 16-shot '{category}': {category}/anomalydino/bank_16-shot.faiss")
    return problems


def available_variants(config: DeploymentConfig, category: str) -> list[str]:
    """Varian AnomalyDINO yang file bank-nya tersedia untuk kategori ini."""
    return [v for v in DINO_VARIANTS if config.bank_path(category, v) is not None]
