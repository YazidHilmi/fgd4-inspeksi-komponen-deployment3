# Inspeksi Visual Mur dan Sekrup (EfficientAD & AnomalyDINO)

Aplikasi Streamlit untuk menguji satu gambar `metal_nut` atau `screw`: status NORMAL/CACAT, skor anomali,
heatmap, dan overlay lokalisasi cacat. Model, preprocessing, dan threshold **sama dengan notebook penelitian**
(`FGD_Kelompok_4_Final_bgt.ipynb`); tidak ada pelatihan ulang dan tidak ada ketergantungan pada Google Colab/Drive.

## 1. Persiapan

Python 3.10–3.12 (anomalib 2.6.2). Dari folder ini:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Jika memakai GPU NVIDIA, pasang PyTorch versi CUDA lebih dulu dari https://pytorch.org, baru `pip install -r requirements.txt`.
Tanpa GPU aplikasi tetap berjalan di CPU (lebih lambat).

## 2. Struktur folder `models/`

Salin hasil notebook (folder `FGD4_Dua_Kategori_Baseline_50_Epoch` di Drive, dari file ZIP) ke folder `models/`
dengan struktur yang sama:

```
models/
├─ deployment_config.json            # threshold, masking, ukuran input (WAJIB)
├─ dinov2_vits14_state_dict.pt       # bobot DINOv2 ViT-S/14 (WAJIB)
├─ screw/
│  ├─ efficientad/efficientad_final_60epoch.ckpt     (WAJIB)
│  └─ anomalydino/bank_16-shot.faiss                 (WAJIB, dipakai juga validasi kategori)
│     anomalydino/bank_1-shot.faiss, bank_full-shot.faiss   (opsional)
└─ metal_nut/
   ├─ efficientad/efficientad_final_60epoch.ckpt     (WAJIB)
   └─ anomalydino/bank_16-shot.faiss                 (WAJIB)
      anomalydino/bank_1-shot.faiss, bank_full-shot.faiss   (opsional)
```

* Path di `deployment_config.json` masih path Colab; aplikasi mencari berkasnya berdasarkan **nama file** di `models/`,
  jadi tidak perlu diubah.
* Varian AnomalyDINO yang file banknya tidak ada otomatis dinonaktifkan di antarmuka.
* Lokasi `models/` dapat diganti dengan variabel lingkungan `INSPEKSI_MODELS_DIR`.
* Dataset MVTec **tidak** diperlukan untuk menjalankan aplikasi (hanya untuk skrip verifikasi di bawah).

## 3. Menjalankan

```bash
streamlit run app.py
```

Buka alamat yang tampil (biasanya http://localhost:8501). Pilih kategori dan model, unggah satu gambar, klik
**Jalankan inspeksi**. Model dimuat sekali per sesi server; inspeksi pertama lebih lama karena pemuatan.

## 4. Apa yang dilakukan aplikasi

| Langkah | Penjelasan |
|---|---|
| Validasi kategori | Patch gambar diekstraksi dengan DINOv2 lalu dicocokkan ke memory bank normal (16-shot) **kedua** kategori (jarak kosinus). Bila kategori yang lebih mirip berbeda dari pilihan pengguna, tampil pesan *"Gambar yang Anda masukkan tidak sesuai dengan kategori yang dipilih."* dan inferensi dihentikan. Nama file tidak dipakai. |
| Penjaga objek asing | Sebelum membandingkan kategori, aplikasi memeriksa apakah gambar masih mirip komponen. Persentil ke-90 jarak patch ke bank referensi normal harus ≤ batas, minimal untuk salah satu kategori. Jika terlalu jauh dari **keduanya**, tampil peringatan *"Gambar yang Anda masukkan bukan gambar sekrup (screw) atau mur (metal nut)."* dan inferensi dihentikan. Batas dihitung otomatis dari bank 16-shot (tiap gambar referensi diuji terhadap referensi lain, hasil terburuk × 2). Untuk batas yang lebih akurat, jalankan skrip kalibrasi (bagian 5b). |
| EfficientAD | Gambar RGB → resize 256×256 (bilinear, antialias) → `EfficientAd` Anomalib 2.6.2 (checkpoint 60 epoch). Skor = maksimum anomaly map. |
| AnomalyDINO | DINOv2 ViT-S/14, sisi pendek 448, normalisasi ImageNet, mode `masking_only` (screw: masking; metal_nut: tanpa), FAISS k=1, skor = `mean_top1p`, peta 256×256. |
| Threshold | `image_threshold` (P95) dan `pixel_threshold` (P99,5) per kategori dan model dari `deployment_config.json`. Gambar CACAT bila skor > image_threshold. |
| Visualisasi | Heatmap `jet` berskala tetap 0…2×pixel_threshold dengan garis putih di batas; overlay merah (alpha 0,55) pada piksel di atas pixel_threshold. |

Kode AnomalyDINO (commit `b9d1c26`) dan DINOv2 (hanya modul inferensi) disertakan di `vendor/` agar tidak perlu
`git clone` atau unduhan `torch.hub`. Lihat `vendor/VERSIONS.txt`.

## 5. Verifikasi terhadap notebook (disarankan sekali)

Untuk memastikan skor aplikasi sama dengan notebook, bandingkan dengan `prediksi_semua_model.csv`:

```bash
python scripts/verify_against_notebook.py --csv models/prediksi_semua_model.csv --dataset "/path/ke/mvtec_anomaly_detection" --per-group 3
```

Skrip memakai engine yang sama dengan aplikasi dan mencetak selisih skor serta kesamaan keputusan.
Selisih sangat kecil (≈ 1e-3 atau kurang; perbedaan GPU/CPU bisa memberi selisih numerik kecil) berarti pipeline sama.

## 5b. Kalibrasi penjaga objek asing (disarankan)

Batas otomatis dibuat tanpa gambar nyata, jadi sebaiknya dikalibrasi sekali memakai dataset dan beberapa foto non-komponen
(mis. 20–50 foto benda lain, tangan, meja, orang). Skrip menulis `models/category_guard.json`, yang otomatis dipakai aplikasi:

```bash
python scripts/calibrate_category_guard.py --dataset "/path/ke/mvtec_anomaly_detection" --negatives "/path/ke/foto_non_komponen"
```

Skrip melaporkan nilai tertinggi gambar komponen (termasuk yang cacat) dan nilai terendah foto negatif, lalu menaruh batas di
antaranya. Jika muncul PERINGATAN (ada foto negatif yang lebih dekat dari komponen), periksa foto tersebut atau naikkan kualitas bank.
Jika komponen asli yang cacat ditolak sebagai "bukan sekrup/mur", jalankan kalibrasi agar batas naik.

## 6. Catatan dan keterbatasan

* Penelitian: MVTec AD, dua kategori, satu seed, 60 epoch EfficientAD; belum divalidasi pada data lini produksi.
* Karena threshold dikalibrasi dari gambar normal, sekitar 5% gambar normal dapat ditandai CACAT secara desain.
* Lokalisasi pada `screw` kurang optimal (cacat sangat kecil), sebagaimana dicatat di notebook.
* Penjaga objek asing berbasis kemiripan fitur terhadap komponen yang dikenal, bukan pengenal objek umum. Objek yang sangat mirip
  sekrup/mur (mis. baut dan ring lain) atau foto yang didominasi latar polos dengan objek kecil bisa lolos; kalibrasi dengan foto negatif
  yang representatif meningkatkan ketepatannya.
* Gambar sebaiknya diambil dengan kondisi serupa dataset (latar polos, objek di tengah).

## 7. Pemecahan masalah

| Gejala | Penyebab / solusi |
|---|---|
| Halaman “Berkas model belum lengkap” | Lengkapi berkas yang tercantum di `models/`, lalu klik **Periksa ulang**. |
| `Checkpoint EfficientAD ... tidak memuat kuantil` | Checkpoint tidak menyimpan kalibrasi peta; kuantil perlu dihitung ulang dari validasi normal (notebook sel 34). |
| `Pustaka belum terpasang` | Jalankan `pip install -r requirements.txt` di environment yang sama dengan `streamlit`. |
| Inspeksi pertama lambat / memori besar | Bank full-shot besar (±300–400 MB per kategori); pilih 16-shot bila memori terbatas. |
