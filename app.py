# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
import os
import streamlit.components.v1 as components
from core.classical import diffract, edges
from core.quantum import build, run, to_phase
from core.reconstruct import enhance

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_BG = os.path.join(BASE_DIR, 'core', 'assets', 'Body.jpg')

try:
    bg_base64 = get_base64_of_bin_file(PATH_TO_BG)
    default_img_src = f'data:image/jpeg;base64,{bg_base64}'
except Exception:
    bg_base64 = ''
    default_img_src = ''

st.set_page_config(page_title="FossilFrame", layout="wide", page_icon="🔬")

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Montserrat:wght@400;600&family=Inter:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stAppViewContainer"], .stApp {
    background: #050505 !important;
    overflow: visible !important;
}
[data-testid="stVerticalBlock"] {
    overflow: visible !important;
}
[data-testid="block-container"] {
    padding: 160px 40px 3rem 40px !important;
    max-width: 100% !important;
    margin: 0 !important;
    background: transparent !important;
    overflow: visible !important;
}

.ff-header {
    position: fixed;
    top: 0; left: 0;
    width: 100%;
    z-index: 1000;
    background: #050505;
    border-bottom: 1px solid #1e1e1e;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 40px;
}
.ff-header-left {
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    z-index: 1001;
}
.main-title {
    font-family: 'Oswald', sans-serif !important;
    font-weight: 700 !important;
    font-size: 5rem !important;
    text-transform: uppercase;
    letter-spacing: 4px;
    color: #e8d070 !important;
    line-height: 1 !important;
    text-shadow: 0 4px 20px rgba(0,0,0,0.9);
    margin: 0 !important;
    padding: 0 !important;
}
.sub-title {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 4px;
    color: #a0a0a0 !important;
    text-transform: uppercase;
    margin: 4px 0 0 0 !important;
}

#spino-wrapper {
    position: fixed;
    top: 50%;
    right: 0;
    transform: translateY(-50%);
    width: 65vw;
    opacity: 0.8;
    z-index: 9999;
    pointer-events: none;
    overflow: visible;
    transition: all 1.2s cubic-bezier(0.4, 0, 0.2, 1);
}
#spino-wrapper img {
    width: 100%;
    height: auto;
    object-fit: contain;
    object-position: right center;
    display: block;
    filter: brightness(1.2) contrast(1.1);
}
#spino-wrapper.logo-mode {
    top: 10px;
    right: 40px;
    transform: translateY(0);
    width: 120px;
    height: 120px;
    opacity: 0.3;
    z-index: 1001;
}

h1, h2, h3, h4, p, label, div, span, li {
    font-family: 'Inter', sans-serif !important;
    color: #e0e0e0 !important;
}
h3, h4 {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    color: #e0e0e0 !important;
}

[data-testid="stImage"], [data-testid="stArrowChart"] {
    background-color: #050505 !important;
    padding: 15px;
    border: 1px solid #d4af37;
    border-radius: 8px;
    box-shadow: 0 0 20px rgba(212,175,55,0.35), 0 10px 30px rgba(0,0,0,0.9);
    position: relative;
    z-index: 100;
    opacity: 1 !important;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 200px;
}
[data-testid="stImage"] img {
    opacity: 1 !important;
    position: relative;
    z-index: 101;
    max-width: 100% !important;
    height: auto !important;
}

.stButton>button {
    background: rgba(17,17,17,0.85);
    border: 1px solid #d4af37;
    color: #d4af37;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 300;
    border-radius: 4px;
    padding: 0.8rem 2rem;
    width: 100%;
    transition: all 0.3s ease;
}
.stButton>button:hover, .stButton>button:active {
    background: rgba(26,26,26,0.95);
    color: #e8d070;
    border-color: #e8d070;
    box-shadow: 0 0 10px rgba(212,175,55,0.2);
}

[data-testid="stFileUploader"] {
    background-color: rgba(17,17,17,0.85) !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px;
    padding: 1rem;
}
.stSelectbox>div>div {
    background-color: rgba(17,17,17,0.85) !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px;
    color: #d4af37;
}

.stSlider>div>div>div { background: #d4af37; box-shadow: none; }
div[role="slider"] {
    background: #111111 !important;
    border: 1px solid #d4af37 !important;
    box-shadow: none !important;
    width: 8px !important;
    height: 16px !important;
    border-radius: 2px !important;
    transform: translateY(-4px);
}
div[data-testid="stSliderTickBar"] {
    background: #333333 !important;
    height: 1px !important;
    margin-top: 5px;
    border-bottom: none;
}

[data-testid="stMetric"] {
    background-color: rgba(17,17,17,0.85);
    border: 1px solid #2a2a2a;
    border-left: 2px solid #d4af37;
    border-radius: 4px;
    padding: 1rem;
}
.metric-box {
    background: rgba(17,17,17,0.85);
    border: 1px solid #2a2a2a;
    border-top: 2px solid #d4af37;
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 400; color: #d4af37 !important; }
.metric-lbl { font-size: 0.75rem; color: #8c8c8c !important; letter-spacing: 2px; text-transform: uppercase; }
[data-testid="stDivider"] { border-color: #2a2a2a; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown("""
<div class="ff-header">
    <div class="ff-header-left">
        <div class="main-title">FOSSIL FRAME</div>
        <div class="sub-title">Quantum-Inspired Virtual Microscope</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_controls, col_content = st.columns([1, 2.5], gap="large")

with col_controls:
    f = st.file_uploader("Upload Fossil Image", type=["png","jpg","jpeg","tiff"])

    sig = st.slider("Diffraction σ", 1, 10, 4)
    dep = st.slider("Circuit Depth", 1, 6, 3)
    alpha = st.slider("Fusion Strength", 0.1, 1.0, 0.5)
    qb = st.selectbox("Qubits", [4, 8, 16], index=1)
    shots = st.selectbox("Shots", [1024, 4096, 8192], index=2)
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("RUN ANALYSIS", use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Upload a fossil image to begin.")
    st.markdown("""
**Pipeline:**  
`Upload -> Simulation -> Phase Map -> Reconstruction`
""")

force_dock = 'true' if f is not None else 'false'

components.html(f"""
<script>
(function() {{
    var forceDock = {force_dock};
    var doc = window.parent.document;

    if (!doc.getElementById('spino-wrapper')) {{
        var wrapper = doc.createElement('div');
        wrapper.id = 'spino-wrapper';
        var img = doc.createElement('img');
        img.id = 'spino-img';
        img.src = '{default_img_src}';
        img.alt = 'fossil';
        wrapper.appendChild(img);
        doc.body.appendChild(wrapper);
    }}

    function apply() {{
        var el = doc.getElementById('spino-wrapper');
        if (!el) return;
        if (forceDock || window.parent.scrollY > 50) {{
            el.classList.add('logo-mode');
        }} else {{
            el.classList.remove('logo-mode');
        }}
    }}

    apply();
    window.parent.removeEventListener('scroll', window.__spinoScroll);
    window.__spinoScroll = apply;
    window.parent.addEventListener('scroll', apply, {{ passive: true }});
}})();
</script>
""", height=0, width=0)

if f is not None:
    with col_content:
        raw = Image.open(f).convert("RGB")
        img = np.array(raw)
        img = cv2.resize(img, (256, 256))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### CLASSICAL OUTPUT")
            blurred = diffract(img, sig)
            st.image(blurred, use_container_width=True, clamp=True)
            st.caption(f"Diffraction blur σ={sig}")

        if run_btn:
            with st.spinner("Executing Qiskit simulation..."):
                qc = build(qb, dep)
                counts = run(qc, shots)
                h, w = blurred.shape
                phase = to_phase(counts, h, w)

            with col2:
                st.markdown("#### QUANTUM PHASE MAP")
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#050505")
                ax.set_facecolor("#050505")
                ax.imshow(phase, cmap="copper")
                ax.axis("off")
                st.pyplot(fig)
                st.caption(f"AerSim · {qb}q · Depth {dep} · {shots} shots")

            with col3:
                st.markdown("#### RECONSTRUCTED")
                out = enhance(blurred, phase, alpha)
                st.image(out, use_container_width=True, clamp=True)
                st.caption("Phase-fused sharpening applied")

            st.divider()
            st.markdown("### PIPELINE METRICS")

            from skimage.metrics import peak_signal_noise_ratio as psnr
            from skimage.metrics import structural_similarity as ssim

            psnr_c = psnr(blurred, blurred)
            psnr_q = psnr(blurred, out)
            ssim_c = ssim(blurred, blurred)
            ssim_q = ssim(blurred, out)
            psnr_c_safe = min(psnr_c, 100) if np.isfinite(psnr_c) else 100
            psnr_q_display = psnr_q if np.isfinite(psnr_q) else 0.0
            gain = ((psnr_q_display - psnr_c_safe) / max(psnr_c_safe, 1)) * 100

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{psnr_q:.1f}</div><div class="metric-lbl">PSNR (dB)</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{ssim_q:.2f}</div><div class="metric-lbl">SSIM</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{gain:.1f}%</div><div class="metric-lbl">GAIN</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{shots}</div><div class="metric-lbl">SHOTS</div></div>', unsafe_allow_html=True)
            with m5:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{qb}q·d{dep}</div><div class="metric-lbl">CIRCUIT</div></div>', unsafe_allow_html=True)

            with col_content:
                st.divider()
                st.markdown("### QISKIT CIRCUIT")
                fig2, ax2 = plt.subplots(figsize=(12, 3))
                fig2.patch.set_facecolor("#050505")
                qc_draw = build(qb, dep)
                qc_draw.draw(output="mpl", ax=ax2, style={"backgroundcolor":"#050505"})
                st.pyplot(fig2)