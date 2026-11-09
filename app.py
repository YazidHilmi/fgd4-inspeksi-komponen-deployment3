"""Aplikasi Streamlit: Inspeksi Visual Mur dan Sekrup (EfficientAD & AnomalyDINO).

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

from inspection import config as cfg
from inspection import ui
from inspection.engine import InspectionEngine, InspectionError, load_rgb
from inspection.render import heatmap_image, original_image, overlay_image, summarize_defect

MISMATCH_MESSAGE = "Gambar yang Anda masukkan tidak sesuai dengan kategori yang dipilih."
FOREIGN_MESSAGE = "Gambar yang Anda masukkan bukan gambar sekrup (screw) atau mur (metal nut)."
MIN_SIDE, MAX_PIXELS = 64, 40_000_000

st.set_page_config(page_title="Inspeksi Visual Mur dan Sekrup", page_icon="🔩", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ util tampilan
def show_image(image, caption: str | None = None) -> None:
    """st.image lintas versi Streamlit (parameter lebar berubah antar versi)."""
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        st.image(image, caption=caption, use_container_width=True)


def step(number: int, title: str, description: str) -> None:
    st.markdown(ui.step_header(number, title, description), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine(_config: cfg.DeploymentConfig, stamp: float) -> InspectionEngine:
    """Satu engine untuk seluruh sesi: checkpoint, bank, dan backbone dimuat sekali (lazy)."""
    return InspectionEngine(_config)


def setup_panel(message: str, problems: list[str]) -> None:
    st.error(message)
    st.markdown("**Berkas yang perlu dilengkapi di folder `models/`:**")
    st.markdown('<ul class="setup">' + "".join(f"<li>{p}</li>" for p in problems) + "</ul>", unsafe_allow_html=True)
    st.caption(f"Folder yang dibaca aplikasi: `{cfg.MODELS_DIR}`. Lihat README.md bagian “Struktur folder models/”.")
    if st.button("Periksa ulang"):
        cfg._files_by_name.cache_clear()
        st.rerun()
    st.stop()


# ------------------------------------------------------------------ header
st.markdown(ui.hero(), unsafe_allow_html=True)

try:
    config = cfg.load_config()
except cfg.SetupError as exc:
    setup_panel(str(exc), exc.problems)
missing = cfg.check_artifacts(config)
if missing:
    setup_panel("Berkas model belum lengkap.", missing)
config_stamp = (cfg.MODELS_DIR / cfg.CONFIG_FILENAME).stat().st_mtime if (cfg.MODELS_DIR / cfg.CONFIG_FILENAME).is_file() else 0.0
engine = get_engine(config, config_stamp)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Petunjuk penggunaan")
    st.markdown(
        "1. Pilih **kategori** komponen dan **model**.\n"
        "2. Unggah **satu gambar** (PNG/JPG).\n"
        "3. Klik **Jalankan inspeksi**.\n"
        "4. Baca hasil: status NORMAL/CACAT, skor, serta heatmap dan overlay area cacat."
    )
    st.markdown("### Tentang metode")
    st.caption(
        "**EfficientAD**: teacher–student yang dilatih pada gambar normal; skor = nilai maksimum peta anomali.\n\n"
        "**AnomalyDINO**: fitur patch DINOv2 dibandingkan dengan memory bank gambar normal; "
        "skor = rata-rata 1% patch paling anomali."
    )
    st.caption(
        "Threshold dikalibrasi dari gambar validasi normal (persentil 95 untuk keputusan gambar, 99,5 untuk piksel). "
        "Akibatnya sebagian kecil gambar normal dapat ditandai cacat secara desain."
    )
    st.caption("Penelitian: MVTec AD, dua kategori, satu seed. Hasil belum divalidasi pada data lini produksi nyata.")

# ------------------------------------------------------------------ langkah 1: input
with st.container(border=True):
    step(1, "Input", "Pilih kategori dan model, lalu unggah gambar yang akan diinspeksi.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        category = st.radio(
            "Kategori komponen",
            cfg.CATEGORIES,
            format_func=lambda c: cfg.CATEGORY_LABELS[c],
            horizontal=True,
        )
        family = st.radio("Model", (cfg.MODEL_EFFICIENTAD, cfg.MODEL_ANOMALYDINO), horizontal=True)
        model_name = cfg.MODEL_EFFICIENTAD
        if family == cfg.MODEL_ANOMALYDINO:
            variants = cfg.available_variants(config, category)
            if not variants:
                st.warning("Memory bank AnomalyDINO untuk kategori ini belum ada di folder models/.")
                st.stop()
            model_name = st.selectbox(
                "Konfigurasi referensi AnomalyDINO",
                variants,
                format_func=lambda v: cfg.DINO_VARIANT_LABELS[v],
                help="Jumlah gambar normal yang dipakai sebagai memory bank (sesuai skenario penelitian).",
            )
        uploaded = st.file_uploader("Unggah satu gambar", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=False)
    image_rgb = None
    with right:
        if uploaded is not None:
            try:
                image_rgb = load_rgb(uploaded)
                h, w = image_rgb.shape[:2]
                if min(h, w) < MIN_SIDE or h * w > MAX_PIXELS:
                    st.error(f"Ukuran gambar tidak didukung ({w}×{h}px). Gunakan sisi terpendek ≥ {MIN_SIDE}px dan maksimum 40 megapiksel.")
                    image_rgb = None
                else:
                    show_image(original_image(image_rgb, 640), caption=f"{uploaded.name} · {w}×{h}px")
            except Exception:  # file rusak / bukan gambar
                st.error("Berkas tidak dapat dibaca sebagai gambar. Unggah PNG atau JPG yang valid.")
        else:
            st.markdown('<div class="note">Belum ada gambar. Unggah gambar komponen untuk memulai.</div>', unsafe_allow_html=True)
    run = st.button("Jalankan inspeksi", type="primary", disabled=image_rgb is None)

signature = (uploaded.name, uploaded.size) if uploaded is not None else None
if st.session_state.get("signature") != signature:  # gambar berganti -> hasil lama dibuang
    st.session_state.pop("result", None)
    st.session_state["signature"] = signature

# ------------------------------------------------------------------ langkah 2: proses
with st.container(border=True):
    step(2, "Proses inspeksi", "Aplikasi memeriksa kategori gambar lebih dulu, lalu menjalankan model.")
    if run and image_rgb is not None:
        st.session_state.pop("result", None)
        result: dict = {"category": category, "model_name": model_name, "file": uploaded.name}
        with st.status("Menjalankan inspeksi…", expanded=True) as status:
            try:
                t0 = time.perf_counter()
                st.write("① Memuat model dan memory bank (hanya sekali per sesi)…")
                engine.preload(category, model_name)
                st.write("② Memeriksa kategori gambar secara visual…")
                check = engine.check_category(image_rgb, category)
                result["check"] = check
                if check.status == "foreign":
                    result["status"] = "foreign"
                    status.update(label="Inspeksi dihentikan: gambar bukan sekrup atau mur", state="error", expanded=False)
                elif check.status == "mismatch":
                    result["status"] = "mismatch"
                    status.update(label="Inspeksi dihentikan: kategori gambar tidak sesuai", state="error", expanded=False)
                else:
                    st.write(f"③ Kategori sesuai. Menjalankan inferensi {model_name}…")
                    output = engine.run(image_rgb, category, model_name)
                    st.write("④ Menyusun heatmap dan overlay lokalisasi…")
                    result.update(
                        status="ok",
                        output=output,
                        summary=summarize_defect(output.anomaly_map, output.pixel_threshold),
                        images={
                            "original": original_image(image_rgb),
                            "heatmap": heatmap_image(output.anomaly_map, output.pixel_threshold),
                            "overlay": overlay_image(image_rgb, output.anomaly_map, output.pixel_threshold),
                        },
                        total_seconds=time.perf_counter() - t0,
                    )
                    status.update(label="Inspeksi selesai", state="complete", expanded=False)
            except (cfg.SetupError, InspectionError) as exc:
                result["status"] = "error"
                result["message"] = str(exc)
                status.update(label="Inspeksi gagal", state="error", expanded=True)
            except Exception as exc:  # kegagalan tak terduga (mis. berkas model rusak)
                result["status"] = "error"
                result["message"] = f"Terjadi kesalahan tak terduga: {type(exc).__name__}: {exc}"
                status.update(label="Inspeksi gagal", state="error", expanded=True)
        st.session_state["result"] = result
    elif "result" in st.session_state:
        st.caption("Inspeksi untuk gambar ini sudah selesai. Hasilnya ada di bawah.")
    else:
        st.caption("Alur: Validasi kategori → Inferensi model → Lokalisasi area cacat. Klik “Jalankan inspeksi” untuk memulai.")

# ------------------------------------------------------------------ langkah 3: hasil
with st.container(border=True):
    step(3, "Hasil deteksi", "Status komponen, skor anomali, dan area yang terdeteksi.")
    result = st.session_state.get("result")
    if result is None:
        st.caption("Hasil inspeksi akan tampil di sini.")
    elif result["status"] == "error":
        st.error(result["message"])
    elif result["status"] == "foreign":
        check = result["check"]
        st.warning(FOREIGN_MESSAGE)
        st.caption(
            "Aplikasi ini hanya dapat memeriksa komponen screw dan metal_nut. Inferensi tidak dilanjutkan karena hasilnya "
            "tidak bermakna untuk objek lain. Unggah gambar sekrup atau mur, lalu jalankan inspeksi lagi."
        )
        with st.expander("Detail pemeriksaan objek"):
            for c in cfg.CATEGORIES:
                st.write(
                    f"{cfg.CATEGORY_LABELS[c]}: jarak patch (P90) = {check.p90[c]:.4f} "
                    f"> batas {check.guards[c]:.4f} (batas {check.guard_source})"
                )
    elif result["status"] == "mismatch":
        check = result["check"]
        st.error(MISMATCH_MESSAGE)
        st.caption(
            f"Kategori dipilih: {cfg.CATEGORY_LABELS[check.selected]}. Pemeriksaan visual menunjukkan gambar lebih mirip "
            f"{cfg.CATEGORY_LABELS[check.predicted]}. Inferensi tidak dilanjutkan; pilih kategori yang sesuai atau unggah gambar lain."
        )
        with st.expander("Detail pemeriksaan kategori"):
            for c in cfg.CATEGORIES:
                st.write(f"{cfg.CATEGORY_LABELS[c]}: rata-rata jarak patch ke referensi normal = {check.distances[c]:.4f}")
    else:
        out, summary, images, check = result["output"], result["summary"], result["images"], result["check"]
        label = cfg.CATEGORY_LABELS[out.category]
        st.markdown(ui.verdict_card(out.is_defect, label), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Skor anomali", f"{out.image_score:.4f}")
        c2.metric("Batas keputusan", f"{out.image_threshold:.4f}")
        c3.metric("Area di atas batas piksel", f"{summary.area_percent:.2f}%", f"{summary.region_count} area", delta_color="off")
        c4.metric("Waktu inferensi", f"{out.inference_ms:.0f} ms")
        st.markdown(ui.score_gauge(out.image_score, out.image_threshold, out.is_defect), unsafe_allow_html=True)
        st.write("")
        a, b, c = st.columns(3, gap="medium")
        with a:
            st.markdown(ui.image_caption("Gambar asli", f"{label} · {result['file']}"), unsafe_allow_html=True)
            show_image(images["original"])
        with b:
            st.markdown(ui.image_caption("Heatmap anomali", "Warna merah = semakin anomali"), unsafe_allow_html=True)
            show_image(images["heatmap"])
            st.markdown(ui.heatmap_legend(), unsafe_allow_html=True)
        with c:
            st.markdown(ui.image_caption("Overlay lokalisasi cacat", "Merah = area di atas batas piksel"), unsafe_allow_html=True)
            show_image(images["overlay"])
        if out.is_defect and summary.region_count == 0:
            st.markdown('<div class="note" style="margin-top:.8rem">Skor gambar melewati batas, tetapi tidak ada piksel yang melewati batas piksel. '
                        'Keduanya dikalibrasi terpisah, sehingga area lokalisasi bisa kosong; keputusan mengikuti skor gambar.</div>', unsafe_allow_html=True)
        elif not out.is_defect and summary.region_count > 0:
            st.markdown('<div class="note" style="margin-top:.8rem">Skor gambar di bawah batas keputusan (NORMAL), walau ada area kecil yang melewati '
                        'batas piksel. Keputusan mengikuti skor gambar; area tersebut dapat berupa derau.</div>', unsafe_allow_html=True)
        with st.expander("Detail teknis"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(
                    f"- **Model**: {out.model_name}\n"
                    f"- **Kategori**: {out.category}\n"
                    f"- **Image threshold** (P95 validasi normal): `{out.image_threshold:.6f}`\n"
                    f"- **Pixel threshold** (P99,5 validasi normal): `{out.pixel_threshold:.6f}`\n"
                    f"- **Skor / batas**: {out.image_score / out.image_threshold:.2f}×\n"
                    f"- **Puncak peta / batas piksel**: {summary.max_ratio:.2f}×"
                )
            with d2:
                st.markdown(
                    f"- **Waktu inferensi**: {out.inference_ms:.1f} ms (tanpa pemuatan model)\n"
                    f"- **Perangkat**: {engine.device}\n"
                    f"- **Ukuran peta anomali**: {config.map_size}×{config.map_size}\n"
                    f"- **Validasi kategori**: jarak ke referensi "
                    + ", ".join(f"{c}={check.distances[c]:.3f}" for c in cfg.CATEGORIES)
                    + (
                        f"\n- **Penjaga objek asing** (batas {check.guard_source}): P90 "
                        + ", ".join(f"{c}={check.p90[c]:.3f}/{check.guards[c]:.3f}" for c in cfg.CATEGORIES)
                        if check.guards
                        else ""
                    )
                )
            st.caption(
                "Skor dan peta dihitung pada resolusi penelitian (peta 256×256). Heatmap memakai skala warna tetap "
                "0 sampai 2× batas piksel sehingga gambar normal tidak tampak “penuh cacat”."
            )
