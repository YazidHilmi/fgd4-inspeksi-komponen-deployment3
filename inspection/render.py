"""Visualisasi hasil inspeksi, mengikuti aturan notebook (sel 43).

* Heatmap: colormap ``jet`` dengan skala warna TETAP 0 .. 2 x pixel threshold, garis putih
  pada batas pixel threshold.
* Overlay: piksel dengan anomaly map > pixel threshold diwarnai merah (alpha 0,55).

Prediksi (mask, skor, keputusan) selalu dihitung pada anomaly map 256x256 seperti notebook.
Untuk tampilan, gambar dan peta hanya diperbesar secara visual (interpolasi bilinear).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib import colormaps
from scipy import ndimage

from .config import HEATMAP_MAX_RATIO, OVERLAY_ALPHA


def _upscale_map(anomaly_map: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(anomaly_map.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def _upscale_image(image_rgb: np.ndarray, size: int) -> np.ndarray:
    # Notebook memakai PIL .resize((256, 256)) untuk overlay; di sini ukuran tampilan lebih besar.
    interpolation = cv2.INTER_AREA if max(image_rgb.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(image_rgb, (size, size), interpolation=interpolation)


def heatmap_image(anomaly_map: np.ndarray, pixel_threshold: float, size: int = 512) -> np.ndarray:
    """Heatmap ``jet`` berskala tetap (0 .. 2 x pixel threshold) dengan kontur putih threshold."""
    shown = _upscale_map(anomaly_map, size)
    scaled = np.clip(shown / (HEATMAP_MAX_RATIO * pixel_threshold), 0.0, 1.0)
    rgb = (colormaps["jet"](scaled)[..., :3] * 255).astype(np.uint8)
    mask = (shown > pixel_threshold).astype(np.uint8)
    if mask.any():
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return rgb


def overlay_image(image_rgb: np.ndarray, anomaly_map: np.ndarray, pixel_threshold: float, size: int = 512) -> np.ndarray:
    """Gambar asli dengan area di atas pixel threshold diwarnai merah (alpha 0,55)."""
    base = _upscale_image(image_rgb, size).astype(np.float32)
    mask = _upscale_map(anomaly_map, size) > pixel_threshold
    red = np.array([255, 0, 0], dtype=np.float32)
    base[mask] = base[mask] * (1.0 - OVERLAY_ALPHA) + red * OVERLAY_ALPHA
    return np.clip(base, 0, 255).astype(np.uint8)


def original_image(image_rgb: np.ndarray, size: int = 512) -> np.ndarray:
    return _upscale_image(image_rgb, size)


@dataclass(frozen=True)
class DefectSummary:
    area_percent: float   # % piksel peta (256x256) yang berada di atas pixel threshold
    region_count: int     # jumlah area terhubung di atas pixel threshold
    max_ratio: float      # nilai maksimum anomaly map dibagi pixel threshold


def summarize_defect(anomaly_map: np.ndarray, pixel_threshold: float) -> DefectSummary:
    mask = anomaly_map > pixel_threshold
    _, count = ndimage.label(mask)
    return DefectSummary(
        area_percent=float(100.0 * mask.mean()),
        region_count=int(count),
        max_ratio=float(anomaly_map.max() / pixel_threshold),
    )


def jet_gradient_css(steps: int = 9) -> str:
    """Gradien CSS jet untuk legenda heatmap (kiri = 0, tengah = threshold, kanan = 2 x threshold)."""
    cmap = colormaps["jet"]
    stops = []
    for i in range(steps):
        r, g, b, _ = cmap(i / (steps - 1))
        stops.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}) {100 * i / (steps - 1):.0f}%")
    return "linear-gradient(90deg," + ",".join(stops) + ")"
