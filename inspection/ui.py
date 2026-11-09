"""Komponen tampilan (CSS + potongan HTML) untuk aplikasi Streamlit."""
from __future__ import annotations

import html

from .render import jet_gradient_css

CSS = """
<style>
:root{
  --ink:#1F2A37; --muted:#667085; --line:#DFE5EC; --surface:#FFFFFF; --bg:#F4F6F9;
  --accent:#2F6F73; --accent-soft:#E4F0F0;
  --ok:#2F7D5B; --ok-soft:#E6F4EC; --bad:#B3453D; --bad-soft:#FBECEA; --warn:#A8731A; --warn-soft:#FDF3E0;
}
html, body, [class*="css"], .stApp{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}
.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1180px;}
#MainMenu, footer{visibility:hidden;}
h1,h2,h3{letter-spacing:-0.01em;}

.hero{background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:1.5rem 1.75rem; margin-bottom:1.25rem;}
.hero .eyebrow{font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); font-weight:600;}
.hero h1{font-size:1.85rem; line-height:1.25; margin:.25rem 0 .5rem 0; color:var(--ink); padding:0;}
.hero p{color:var(--muted); font-size:.98rem; margin:0; max-width:62ch;}
.chips{display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem;}
.chip{background:var(--accent-soft); color:var(--accent); border-radius:999px; padding:.18rem .7rem; font-size:.78rem; font-weight:600;}

.step{display:flex; align-items:center; gap:.7rem; margin-bottom:.35rem;}
.step .num{width:1.7rem; height:1.7rem; border-radius:50%; background:var(--accent); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:.9rem; flex:none;}
.step .ttl{font-size:1.1rem; font-weight:650; color:var(--ink);}
.step-desc{color:var(--muted); font-size:.9rem; margin:0 0 .9rem 2.4rem;}

.verdict{border-radius:14px; padding:1rem 1.25rem; border:1px solid; margin-bottom:1rem;}
.verdict .t{font-size:1.35rem; font-weight:700; letter-spacing:.01em;}
.verdict .s{font-size:.93rem; margin-top:.15rem;}
.verdict.ok{background:var(--ok-soft); border-color:#BFE0CF; color:#1E5C42;}
.verdict.bad{background:var(--bad-soft); border-color:#EBC3BF; color:#8A322B;}

.gauge{position:relative; height:10px; background:#E8ECF1; border-radius:999px; margin:.5rem 0 .3rem 0;}
.gauge .fill{position:absolute; left:0; top:0; bottom:0; border-radius:999px;}
.gauge .fill.ok{background:var(--ok);} .gauge .fill.bad{background:var(--bad);}
.gauge .thr{position:absolute; top:-5px; bottom:-5px; width:2px; background:var(--ink);}
.gauge-legend{display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted);}

.legend-bar{height:10px; border-radius:999px; margin-top:.4rem; background:__JET__;}
.legend-labels{display:flex; justify-content:space-between; font-size:.75rem; color:var(--muted); margin-top:.2rem;}

.imgcap{font-weight:600; color:var(--ink); font-size:.95rem; margin-bottom:.25rem;}
.imgsub{color:var(--muted); font-size:.82rem; margin-bottom:.4rem;}
.note{background:var(--warn-soft); border:1px solid #F0DDB5; color:#6F4D10; border-radius:10px; padding:.6rem .85rem; font-size:.88rem;}
.setup li{margin-bottom:.25rem;}
div[data-testid="stMetric"]{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.7rem .9rem;}
div[data-testid="stMetricLabel"]{color:var(--muted);}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;}
</style>
""".replace("__JET__", jet_gradient_css())


def hero() -> str:
    return """
<div class="hero">
  <div class="eyebrow">Inspeksi visual · Deteksi &amp; lokalisasi cacat</div>
  <h1>Inspeksi Visual Mur dan Sekrup</h1>
  <p>Unggah satu gambar komponen, pilih kategori dan model, lalu lihat apakah komponen NORMAL atau CACAT
  beserta area cacat yang terdeteksi. Model hanya mempelajari gambar normal (unsupervised).</p>
  <div class="chips"><span class="chip">EfficientAD</span><span class="chip">AnomalyDINO</span>
  <span class="chip">metal_nut</span><span class="chip">screw</span><span class="chip">MVTec AD</span></div>
</div>"""


def step_header(number: int, title: str, description: str) -> str:
    return (
        f'<div class="step"><div class="num">{number}</div><div class="ttl">{html.escape(title)}</div></div>'
        f'<p class="step-desc">{html.escape(description)}</p>'
    )


def verdict_card(is_defect: bool, category_label: str) -> str:
    if is_defect:
        title, sub, cls = "CACAT TERDETEKSI", f"{category_label} ini sebaiknya ditarik dari lini produksi untuk diperiksa lebih lanjut.", "bad"
    else:
        title, sub, cls = "NORMAL", f"{category_label} ini lolos inspeksi: skor anomali tidak melewati batas keputusan.", "ok"
    return f'<div class="verdict {cls}"><div class="t">{title}</div><div class="s">{html.escape(sub)}</div></div>'


def score_gauge(score: float, threshold: float, is_defect: bool) -> str:
    scale = max(score, threshold) * 1.3
    fill = min(100.0, 100.0 * score / scale)
    thr = min(100.0, 100.0 * threshold / scale)
    cls = "bad" if is_defect else "ok"
    return (
        f'<div class="gauge"><div class="fill {cls}" style="width:{fill:.1f}%"></div>'
        f'<div class="thr" style="left:{thr:.1f}%"></div></div>'
        f'<div class="gauge-legend"><span>0</span><span>Garis hitam = batas keputusan ({threshold:.4f})</span></div>'
    )


def heatmap_legend() -> str:
    return (
        '<div class="legend-bar"></div>'
        '<div class="legend-labels"><span>0</span><span>batas piksel (garis putih)</span><span>&ge; 2&times; batas</span></div>'
    )


def image_caption(title: str, subtitle: str) -> str:
    return f'<div class="imgcap">{html.escape(title)}</div><div class="imgsub">{html.escape(subtitle)}</div>'
