# FossilFrame Application
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from core.classical import diffract, edges
from core.quantum import build, run, to_phase
from core.reconstruct import enhance

st.set_page_config(page_title="FossilFrame", layout="wide", page_icon="🔬")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #07090f;
}
[data-testid="stSidebar"] {
    background-color: #0d1120;
    border-right: 1px solid #1a2540;
}
[data-testid="stHeader"] {
    background-color: #07090f;
}
[data-testid="block-container"] {
    background-color: #07090f;
    padding-top: 2rem;
}
h1, h2, h3, h4, p, label, div, span {
    color: #cdd6f4 !important;
}
.stButton>button {
    background: linear-gradient(135deg, #8b2fff, #5500cc);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #00f0ff, #0088aa);
    color: #07090f;
}
[data-testid="stFileUploader"] {
    background-color: #0d1120;
    border: 1.5px dashed #00f0ff;
    border-radius: 8px;
    padding: 1rem;
}
.stSelectbox>div>div {
    background-color: #0d1120;
    color: #cdd6f4;
    border: 1px solid #1a2540;
}
.stSlider>div>div>div {
    background: #00f0ff;
}
[data-testid="stMetric"] {
    background-color: #0d1120;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 1rem;
}
.metric-box {
    background: #0d1120;
    border: 1px solid #1a2540;
    border-top: 2px solid #00f0ff;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-size: 2rem;
    font-weight: 900;
    color: #00f0ff !important;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #4a5578 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
[data-testid="stDivider"] {
    border-color: #1a2540;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔬 FossilFrame")
st.markdown("**PS04 · Quantum-Inspired Virtual Microscope · QTHack04 · Team VultureOps**")
st.divider()

with st.sidebar:
    st.markdown("### Controls")
    f = st.file_uploader("Upload Fossil Image", type=["png","jpg","jpeg","tiff"])
    sig = st.slider("Diffraction σ", 1, 10, 4)
    dep = st.slider("Circuit Depth", 1, 6, 3)
    qb = st.selectbox("Qubits", [4, 8, 16], index=1)
    shots = st.selectbox("Shots", [1024, 4096, 8192], index=2)
    alpha = st.slider("Fusion Strength", 0.1, 1.0, 0.5)
    run_btn = st.button("▶ RUN ANALYSIS")

if f is not None:
    raw = Image.open(f).convert("RGB")
    img = np.array(raw)
    img = cv2.resize(img, (256, 256))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔴 Classical Output")
        blurred = diffract(img, sig)
        st.image(blurred, width='stretch', clamp=True)
        st.caption(f"Diffraction blur σ={sig} applied · Abbe limit ~200nm")

    if run_btn:
        with st.spinner("Running Qiskit simulation..."):
            qc = build(qb, dep)
            counts = run(qc, shots)
            h, w = blurred.shape
            phase = to_phase(counts, h, w)

        with col2:
            st.markdown("#### 🟣 Quantum Phase Map")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#07090f")
            ax.set_facecolor("#07090f")
            ax.imshow(phase, cmap="plasma")
            ax.axis("off")
            st.pyplot(fig)
            st.caption(f"Qiskit AerSim · {qb} qubits · depth {dep} · {shots} shots")

        with col3:
            st.markdown("#### 🟢 AI Reconstructed")
            out = enhance(blurred, phase, alpha)
            st.image(out, width='stretch', clamp=True)
            st.caption("Phase-fused + sharpening kernel applied")

        st.divider()
        st.markdown("### 📊 Pipeline Metrics")

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
        st.markdown("### ⚛️ Qiskit Circuit")
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        fig2.patch.set_facecolor("#07090f")
        qc_draw = build(qb, dep)
        qc_draw.draw(output="mpl", ax=ax2, style={"backgroundcolor":"#07090f"})
        st.pyplot(fig2)

else:
    st.info("Upload a fossil image in the sidebar to begin.")
    st.markdown("""
    **Pipeline:**  
    `Upload → Diffraction Simulation → Qiskit Circuit → Phase Map → AI Reconstruction → Metrics`
    """)