# -*- coding: utf-8 -*-
# FossilFrame Application
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64
from core.classical import diffract, edges
from core.quantum import build, run, to_phase
from core.reconstruct import enhance

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_BG = os.path.join(BASE_DIR, 'core', 'assets', 'Body.jpg')
PATH_TO_LOGO = os.path.join(BASE_DIR, 'core', 'assets', 'Header.jpg')
# ---------------------

try:
    bg_base64 = get_base64_of_bin_file(PATH_TO_BG)
    bg_css = f'''
background-image: 
    linear-gradient(to right, #050505 0%, #050505 30%, rgba(5,5,5,0.3) 50%, transparent 70%),
    url("data:image/jpeg;base64,{bg_base64}");
''' 
except Exception:
    bg_css = "background-image: none;"

try:
    logo_base64 = get_base64_of_bin_file(PATH_TO_LOGO)
    logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" style="height: 1.1em; object-fit: contain; filter: sepia(1) hue-rotate(15deg) brightness(1.3) contrast(1.1) opacity(0.9); margin-bottom: 8px;">'
except Exception:
    logo_html = ""

st.set_page_config(page_title="FossilFrame", layout="wide", page_icon="🔬")

css_template = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Montserrat:wght@400;600&family=Inter:wght@300;400;600&display=swap');

[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stHeader"] { background-color: transparent; }
[data-testid="block-container"] {
    padding: 2rem 40px 3rem 40px !important;
    max-width: 100% !important;
    margin: 0 !important;
}
.stApp {
    background-color: #050505;
}
.fossil-bg {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -1;
    pointer-events: none;
    __BG_CSS__
    background-size: cover, 80vw auto; 
    background-repeat: no-repeat, no-repeat;
    background-position: left top, right center;
    opacity: 1;
    transform: translateX(0);
    transition: opacity 0.7s ease-in-out, transform 0.7s ease-in-out;
    filter: brightness(1.25) contrast(1.15);
}
.fossil-bg.hidden {
    opacity: 0;
    transform: translateX(100px);
}
h1, h2, h3, h4, p, label, div, span, li {
    font-family: 'Inter', sans-serif !important;
    color: #e0e0e0 !important;
}

.main-title {
    display: flex;
    align-items: center;
    gap: 15px;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 700 !important;
    font-size: 6rem !important;
    text-transform: uppercase;
    letter-spacing: 4px;
    color: #e8d070 !important; /* Cinematic soft gold */
    line-height: 1 !important;
    padding-bottom: 0px !important;
    text-shadow: 0 8px 30px rgba(0, 0, 0, 0.8);
}

.sub-title {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    letter-spacing: 4px;
    color: #a0a0a0 !important;
    text-transform: uppercase;
    margin-bottom: 3rem !important;
}

h3, h4 {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    color: #e0e0e0 !important; /* Minimal clean, slightly white */
}

/* Image Results Containers */
[data-testid="stImage"], [data-testid="stArrowChart"] {
    background-color: #050505 !important;
    padding: 15px;
    border: 1px solid #d4af37;
    border-radius: 8px;
    box-shadow: 0 0 20px rgba(212, 175, 55, 0.4), 0 10px 30px rgba(0, 0, 0, 0.9);
    position: relative;
    z-index: 100;
    opacity: 1 !important;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 200px; /* Fallback height to prevent bounding box collapse */
}
[data-testid="stImage"] img {
    opacity: 1 !important;
    position: relative;
    z-index: 101;
    max-width: 100% !important;
    height: auto !important;
}

/* Button */
.stButton>button {
    background: rgba(17, 17, 17, 0.85);
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
    background: rgba(26, 26, 26, 0.95);
    color: #e8d070;
    border-color: #e8d070;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.2);
}

/* Controls */
[data-testid="stFileUploader"] {
    background-color: rgba(17, 17, 17, 0.85) !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px;
    padding: 1rem;
}
.stSelectbox>div>div {
    background-color: rgba(17, 17, 17, 0.85) !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px;
    color: #d4af37;
}

/* Sliders */
.stSlider>div>div>div {
    background: #d4af37;
    box-shadow: none;
}
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

/* Metrics */
[data-testid="stMetric"] {
    background-color: rgba(17, 17, 17, 0.85);
    border: 1px solid #2a2a2a;
    border-left: 2px solid #d4af37;
    border-radius: 4px;
    padding: 1rem;
}
.metric-box {
    background: rgba(17, 17, 17, 0.85);
    border: 1px solid #2a2a2a;
    border-top: 2px solid #d4af37;
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-size: 2rem;
    font-weight: 400;
    color: #d4af37 !important;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #8c8c8c !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
[data-testid="stDivider"] {
    border-color: #2a2a2a;
}
</style>
"""

st.markdown(f'<div class="main-title">{logo_html}FOSSIL FRAME</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Quantum-Inspired Virtual Microscope</div>', unsafe_allow_html=True)
st.divider()

col_controls, col_content = st.columns([1, 2.5], gap="large")

with col_controls:
    f = st.file_uploader("Upload Fossil Image", type=["png","jpg","jpeg","tiff"])
    
    # Dynamic CSS rendering based on App state
    final_css = css_template.replace("__BG_CSS__", bg_css)
    st.markdown(final_css, unsafe_allow_html=True)
    
    # Mount persistent background DOM node strictly as a static string to prevent React teardown
    st.markdown('<div class="fossil-bg"></div>', unsafe_allow_html=True)
    
    # Trigger CSS transition independently via Javascript to bypass React DOM teardown
    import streamlit.components.v1 as components
    components.html(f"""
    <script>
        const bgs = window.parent.document.querySelectorAll('.fossil-bg');
        bgs.forEach(bg => {{
            if ("{str(f is not None).lower()}" === "true") {{
                bg.classList.add('hidden');
            }} else {{
                bg.classList.remove('hidden');
            }}
        }});
    </script>
    """, height=0, width=0)
    st.markdown("<br>", unsafe_allow_html=True)
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

with col_content:
    if f is not None:
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
                st.image(out, width='stretch', clamp=True)
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

            st.divider()
            st.markdown("### QISKIT CIRCUIT")
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            fig2.patch.set_facecolor("#050505")
            qc_draw = build(qb, dep)
            qc_draw.draw(output="mpl", ax=ax2, style={"backgroundcolor":"#050505"})
            st.pyplot(fig2)