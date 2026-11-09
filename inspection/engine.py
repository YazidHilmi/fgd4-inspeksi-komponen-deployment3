"""Mesin inferensi: memuat model sekali, memvalidasi kategori, dan menjalankan inspeksi.

Seluruh langkah inferensi meniru notebook penelitian:
* ``predict_dino``       -> notebook sel 32 (AnomalyDINO, masking_only, k=1, mean_top1p)
* ``predict_efficientad``-> notebook sel 38 (resize 256x256 bilinear antialias, skor = max peta)
* threshold              -> ``deployment_config.json`` (persentil 95 skor gambar, 99,5 piksel)

Modul ini tidak bergantung pada Streamlit. Impor berat (torch, faiss, anomalib) dilakukan
saat pertama kali dibutuhkan.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .config import (
    CATEGORIES,
    CATEGORY_CHECK_BANK,
    GUARD_FILENAME,
    MODELS_DIR,
    MODEL_EFFICIENTAD,
    OOD_AUTO_MAX_REFERENCES,
    OOD_FACTOR,
    OOD_PERCENTILE,
    VENDOR_DIR,
    DeploymentConfig,
    SetupError,
)


class InspectionError(RuntimeError):
    """Kegagalan saat inferensi (mis. foreground kosong)."""


@dataclass(frozen=True)
class CategoryCheck:
    selected: str
    predicted: str                      # kategori dengan rata-rata jarak patch terkecil
    distances: dict[str, float]         # rata-rata jarak kosinus patch ke bank tiap kategori
    p90: dict[str, float]               # persentil ke-90 jarak patch ke bank tiap kategori
    guards: dict[str, float] | None     # batas maksimum p90 agar dianggap komponen kategori tsb
    guard_source: str                   # "kalibrasi" | "otomatis" | "tidak tersedia"

    @property
    def is_foreign(self) -> bool:
        """True bila gambar terlalu jauh dari KEDUA kategori (bukan sekrup maupun mur)."""
        if not self.guards:
            return False
        return all(self.p90[c] > self.guards[c] for c in CATEGORIES)

    @property
    def status(self) -> str:
        if self.is_foreign:
            return "foreign"
        return "ok" if self.predicted == self.selected else "mismatch"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def guard_from_bank_vectors(vectors: np.ndarray, per_image: int, factor: float, percentile: float, max_images: int) -> float | None:
    """Batas p90 'masih mirip komponen' dari bank referensi saja (tanpa dataset).

    ``vectors``: fitur patch ter-normalisasi L2 (N x D), tersusun per gambar referensi berurutan
    (``per_image`` patch tiap gambar). Setiap gambar referensi diuji terhadap patch gambar referensi
    LAIN (leave-one-image-out); nilai terburuk dikali ``factor`` menjadi batas. Jarak = 1 - cos
    (skala yang sama dengan skor patch AnomalyDINO).
    """
    n = len(vectors)
    images = n // per_image
    if per_image <= 0 or n % per_image or images < 4:
        return None
    worst = 0.0
    for i in range(min(images, max_images)):
        lo, hi = i * per_image, (i + 1) * per_image
        query = vectors[lo:hi]
        rest = np.concatenate([vectors[:lo], vectors[hi:]])
        distance = 1.0 - (query @ rest.T).max(axis=1)
        worst = max(worst, float(np.percentile(distance, percentile)))
    return worst * factor


@dataclass(frozen=True)
class InspectionOutput:
    category: str
    model_name: str
    image_score: float
    image_threshold: float
    pixel_threshold: float
    anomaly_map: np.ndarray       # float32, MAP_SIZE x MAP_SIZE
    inference_ms: float

    @property
    def is_defect(self) -> bool:
        return self.image_score > self.image_threshold


def load_rgb(source) -> np.ndarray:
    """Membaca gambar (path atau file-like) menjadi array RGB uint8.

    Gambar grayscale (mode L, seperti ``screw`` MVTec) dibaca sebagai RGB, sama seperti notebook.
    """
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


class InspectionEngine:
    def __init__(self, config: DeploymentConfig, device: str | None = None):
        self.config = config
        self._device_arg = device
        self._lock = threading.RLock()
        self._rt: dict = {}
        self._dino_model = None
        self._banks: dict[tuple[str, str], object] = {}
        self._effad: dict[str, object] = {}

    # ------------------------------------------------------------------ runtime
    def _runtime(self) -> dict:
        """Impor pustaka berat sekali saja dan siapkan sys.path untuk kode vendor."""
        if self._rt:
            return self._rt
        for path in (VENDOR_DIR / "AnomalyDINO", VENDOR_DIR / "dinov2"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            import faiss
            import torch
            from src.backbones import DINOv2Wrapper
            from src.post_eval import mean_top1p
            from src.utils import dists2map
        except ImportError as exc:  # pragma: no cover - tergantung lingkungan
            raise SetupError(
                f"Pustaka belum terpasang: {exc}. Jalankan 'pip install -r requirements.txt'."
            ) from exc
        self._rt = {
            "torch": torch,
            "faiss": faiss,
            "DINOv2Wrapper": DINOv2Wrapper,
            "mean_top1p": mean_top1p,
            "dists2map": dists2map,
        }
        return self._rt

    @property
    def device(self) -> str:
        torch = self._runtime()["torch"]
        if self._device_arg:
            return self._device_arg
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _sync(self) -> None:
        torch = self._runtime()["torch"]
        if self.device == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ loaders
    def _dino(self):
        """Backbone DINOv2 ViT-S/14 dari ``dinov2_vits14_state_dict.pt`` (tanpa unduhan)."""
        if self._dino_model is not None:
            return self._dino_model
        rt = self._runtime()
        torch = rt["torch"]
        weights = self.config.backbone_weights()
        if weights is None:
            raise SetupError("Bobot DINOv2 (dinov2_vits14_state_dict.pt) tidak ditemukan di folder models/.")
        from torchvision import transforms

        base = rt["DINOv2Wrapper"]
        size = self.config.dino_size
        device = self.device

        class OfflineDINOv2(base):
            """DINOv2Wrapper milik AnomalyDINO, tetapi bobot dibaca dari berkas lokal."""

            def load_model(self_inner):
                from dinov2.models import vision_transformer as vits

                # Argumen sama dengan torch.hub 'dinov2_vits14' (pretrained=False).
                model = vits.vit_small(
                    patch_size=14,
                    img_size=518,
                    init_values=1.0,
                    ffn_layer="mlp",
                    block_chunks=0,
                    num_register_tokens=0,
                    interpolate_antialias=False,
                    interpolate_offset=0.1,
                )
                state = torch.load(weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                model.eval()
                # Preprocessing identik dengan DINOv2Wrapper.load_model bawaan.
                self_inner.transform = transforms.Compose([
                    transforms.Resize(size=self_inner.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                return model.to(self_inner.device)

        self._dino_model = OfflineDINOv2(self.config.dino_backbone, device, size)
        return self._dino_model

    def _bank(self, category: str, model_name: str):
        key = (category, model_name)
        if key not in self._banks:
            path = self.config.bank_path(category, model_name)
            if path is None:
                raise SetupError(f"Memory bank untuk '{category}' / {model_name} tidak ditemukan di folder models/.")
            self._banks[key] = self._runtime()["faiss"].read_index(str(path))
        return self._banks[key]

    def _efficientad(self, category: str):
        if category in self._effad:
            return self._effad[category]
        rt = self._runtime()
        torch = rt["torch"]
        path = self.config.efficientad_checkpoint(category)
        if path is None:
            raise SetupError(f"Checkpoint EfficientAD '{category}' tidak ditemukan di folder models/.")
        try:
            from anomalib.models import EfficientAd
        except ImportError as exc:  # pragma: no cover
            raise SetupError(f"anomalib belum terpasang: {exc}. Jalankan 'pip install -r requirements.txt'.") from exc

        # Sama dengan notebook sel 34: post_processor=False agar skor tetap kontinu.
        model = EfficientAd(post_processor=False, evaluator=False, visualizer=False)
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        result = model.load_state_dict(state, strict=False)
        optional = ("post_processor.", "evaluator.", "visualizer.")
        unexpected = [k for k in result.unexpected_keys if not k.startswith(optional)]
        missing = [k for k in result.missing_keys if not k.startswith(optional)]
        if unexpected or missing:
            raise SetupError(f"Checkpoint EfficientAD '{category}' tidak cocok: missing={missing}, unexpected={unexpected}")
        model = model.to(self.device).eval()
        # Kuantil normalisasi peta dan statistik teacher ikut tersimpan di checkpoint
        # (ParameterDict pada EfficientAdModel). Tanpa keduanya skor tidak sama dengan penelitian.
        net = model.model
        if not net.is_set(net.quantiles) or not net.is_set(net.mean_std):
            raise SetupError(
                f"Checkpoint EfficientAD '{category}' tidak memuat kuantil/mean_std hasil kalibrasi. "
                "Hitung ulang dari validasi normal seperti notebook sel 34 (membutuhkan split_*.csv dan gambar validasi)."
            )
        self._effad[category] = model
        return model

    def preload(self, category: str, model_name: str) -> None:
        """Memuat semua yang dibutuhkan agar inferensi pertama tidak lambat."""
        with self._lock:
            self._dino()
            for cat in CATEGORIES:
                self._bank(cat, CATEGORY_CHECK_BANK)
            self._guards()
            if model_name == MODEL_EFFICIENTAD:
                self._efficientad(category)
            else:
                self._bank(category, model_name)

    # ------------------------------------------------------------------ validasi kategori
    def _guards(self) -> tuple[dict[str, float] | None, str]:
        """Batas penjaga 'bukan sekrup/mur': berkas kalibrasi bila ada, jika tidak otomatis dari bank."""
        if hasattr(self, "_guard_cache"):
            return self._guard_cache
        result: tuple[dict[str, float] | None, str] = (None, "tidak tersedia")
        path = MODELS_DIR / GUARD_FILENAME
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                guards = {c: float(data["guards"][c]) for c in CATEGORIES}
                result = (guards, "kalibrasi")
            except (KeyError, ValueError, TypeError):
                result = (None, "tidak tersedia")
        if result[0] is None:
            per_image = (self.config.dino_size // 14) ** 2
            guards = {}
            for category in CATEGORIES:
                bank = self._bank(category, CATEGORY_CHECK_BANK)
                vectors = bank.reconstruct_n(0, bank.ntotal)
                value = guard_from_bank_vectors(vectors, per_image, OOD_FACTOR, OOD_PERCENTILE, OOD_AUTO_MAX_REFERENCES)
                if value is None:
                    guards = {}
                    break
                guards[category] = value
            result = (guards, "otomatis") if guards else (None, "tidak tersedia")
        self._guard_cache = result
        return result

    def category_stats(self, image_rgb: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        """Rata-rata dan persentil ke-90 jarak patch gambar ke bank referensi normal tiap kategori."""
        with self._lock:
            rt = self._runtime()
            faiss = rt["faiss"]
            dino = self._dino()
            tensor, _ = dino.prepare_image(image_rgb)
            patches = dino.extract_features(tensor)
            query = np.ascontiguousarray(patches, dtype=np.float32)
            faiss.normalize_L2(query)
            mean: dict[str, float] = {}
            tail: dict[str, float] = {}
            for category in CATEGORIES:
                d, _ = self._bank(category, CATEGORY_CHECK_BANK).search(query, 1)
                patch = d[:, 0] / 2  # skala sama dengan skor patch AnomalyDINO (= 1 - cos)
                mean[category] = float(np.mean(patch))
                tail[category] = float(np.percentile(patch, OOD_PERCENTILE))
        return mean, tail

    def check_category(self, image_rgb: np.ndarray, selected: str) -> CategoryCheck:
        """Pemeriksaan visual memakai fitur patch DINOv2 (bukan nama file).

        1. Gambar yang terlalu jauh dari bank KEDUA kategori ditandai 'foreign' (bukan sekrup/mur).
        2. Selain itu, kategori dengan rata-rata jarak terkecil dianggap isi gambar; bila berbeda
           dari pilihan pengguna -> 'mismatch'. Pada kedua kasus inferensi tidak dilanjutkan.
        """
        mean, tail = self.category_stats(image_rgb)
        with self._lock:
            guards, source = self._guards()
        predicted = min(mean, key=mean.get)
        return CategoryCheck(selected=selected, predicted=predicted, distances=mean, p90=tail, guards=guards, guard_source=source)

    # ------------------------------------------------------------------ inferensi
    def _predict_dino(self, image_rgb: np.ndarray, category: str, model_name: str) -> tuple[float, np.ndarray, float]:
        rt = self._runtime()
        faiss, dists2map, mean_top1p = rt["faiss"], rt["dists2map"], rt["mean_top1p"]
        dino = self._dino()
        bank = self._bank(category, model_name)
        map_size = self.config.map_size
        self._sync()
        started = time.perf_counter()
        tensor, grid = dino.prepare_image(image_rgb)
        patches = dino.extract_features(tensor)
        foreground = dino.compute_background_mask(patches, grid, threshold=10, masking_type=self.config.masking(category))
        if not foreground.any():
            raise InspectionError("Objek tidak terdeteksi pada gambar (foreground kosong).")
        query = np.ascontiguousarray(patches[foreground], dtype=np.float32)
        faiss.normalize_L2(query)
        distance, _ = bank.search(query, 1)
        patch_scores = np.zeros(len(patches), dtype=np.float32)
        patch_scores[foreground] = distance[:, 0] / 2
        anomaly_map = dists2map(patch_scores.reshape(grid), (map_size, map_size)).astype(np.float32)
        score = float(mean_top1p(patch_scores))
        self._sync()
        return score, anomaly_map, time.perf_counter() - started

    def _predict_efficientad(self, image_rgb: np.ndarray, category: str) -> tuple[float, np.ndarray, float]:
        rt = self._runtime()
        torch = rt["torch"]
        from torchvision.transforms import functional as TF

        model = self._efficientad(category)
        map_size = self.config.map_size
        self._sync()
        started = time.perf_counter()
        tensor = TF.to_tensor(Image.fromarray(image_rgb))
        tensor = TF.resize(tensor, [256, 256], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)[None].to(self.device)
        with torch.inference_mode():
            output = model.model(tensor)
        score = float(output.pred_score.detach().cpu().reshape(-1)[0])
        anomaly_map = output.anomaly_map.detach().float()
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map[:, None]
        anomaly_map = torch.nn.functional.interpolate(anomaly_map, size=(map_size, map_size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        self._sync()
        return score, anomaly_map, time.perf_counter() - started

    def run(self, image_rgb: np.ndarray, category: str, model_name: str) -> InspectionOutput:
        """Inferensi satu gambar dengan model dan threshold penelitian."""
        image_threshold, pixel_threshold = self.config.thresholds(category, model_name)
        with self._lock:
            if model_name == MODEL_EFFICIENTAD:
                score, anomaly_map, seconds = self._predict_efficientad(image_rgb, category)
            else:
                score, anomaly_map, seconds = self._predict_dino(image_rgb, category, model_name)
        if not (np.isfinite(score) and np.isfinite(anomaly_map).all()):
            raise InspectionError("Model menghasilkan nilai tidak valid (NaN/Inf).")
        return InspectionOutput(
            category=category,
            model_name=model_name,
            image_score=score,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            anomaly_map=anomaly_map,
            inference_ms=1000.0 * seconds,
        )
