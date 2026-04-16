import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import gdown
import requests
import os

MODEL_PATH = "hybrid_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://huggingface.co/Sahil200217/pneumonia-detection-model/resolve/main/hybrid_model.h5"
    with requests.get(url, stream=True) as r:
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="centered",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e17;
    color: #e2e8f0;
}

.stApp {
    background: #0a0e17;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0 30%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 0.75rem;
}

.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Architecture Pill Row ── */
.arch-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 2rem 0 2.5rem;
}

.arch-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    color: #94a3b8;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 0 0 2rem;
}

/* ── Upload Zone ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
    display: block;
}

/* Override Streamlit uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.25) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(56, 189, 248, 0.5) !important;
}

[data-testid="stFileUploader"] label {
    color: #64748b !important;
}

/* ── Image Display ── */
[data-testid="stImage"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Result Cards ── */
.result-card {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    padding: 1px;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}

.result-card.pneumonia {
    background: rgba(239, 68, 68, 0.07);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.result-card.normal {
    background: rgba(34, 197, 94, 0.07);
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.result-icon {
    font-size: 2.2rem;
    line-height: 1;
    flex-shrink: 0;
}

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
}

.result-label.pneumonia { color: #f87171; }
.result-label.normal { color: #4ade80; }

.result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.05em;
}

/* ── Confidence Bar ── */
.conf-bar-wrap {
    margin-top: 1.2rem;
}

.conf-bar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.conf-bar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}

.conf-bar-fill-pneumonia {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #dc2626, #f87171);
    transition: width 0.8s ease;
}

.conf-bar-fill-normal {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #16a34a, #4ade80);
    transition: width 0.8s ease;
}

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
}

.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}

.stat-label {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 0.2rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Footer ── */
.footer-section {
    margin-top: 3rem;
    padding: 1.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}

.footer-icon {
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

.footer-text {
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.6;
}

.footer-text strong {
    color: #64748b;
}

/* ── Streamlit cleanup ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 720px; }
[data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pneumonia_model():
    return load_model("hybrid_model.h5")

model = load_pneumonia_model()


# ─── Hero Section ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML Research Project · Chest X-Ray Analysis</div>
    <h1 class="hero-title">PneumoScan AI</h1>
    <p class="hero-subtitle">
        Benchmarked ResNet50, DenseNet121, EfficientNetB0 &amp; DeiT —
        then built a feature fusion hybrid of B0 + DenseNet121
        achieving 89.3% accuracy on chest X-ray classification.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="arch-row">
    <span class="arch-pill">ResNet50</span>
    <span class="arch-pill">DenseNet121</span>
    <span class="arch-pill">EfficientNetB0</span>
    <span class="arch-pill">DeiT</span>
    <span class="arch-pill">Feature Fusion Hybrid</span>
    <span class="arch-pill">TensorFlow · Keras</span>
</div>
""", unsafe_allow_html=True)

# ─── Model Stats ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">89.3%</div>
        <div class="stat-label">Hybrid Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">4 Models</div>
        <div class="stat-label">Benchmarked</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">224×224</div>
        <div class="stat-label">Input Resolution</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">2-class</div>
        <div class="stat-label">Classification</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider' style='margin-top:2rem'>", unsafe_allow_html=True)

# ─── Upload Section ──────────────────────────────────────────────────────────
st.markdown("<span class='upload-label'>Upload Chest X-Ray</span>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Drop a chest X-ray image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ─── Inference ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("<span class='upload-label' style='margin-top:1rem;display:block'>Input Image</span>", unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    with col2:
        st.markdown("<span class='upload-label' style='margin-top:1rem;display:block'>Analysis</span>", unsafe_allow_html=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Running inference..."):
            prediction = model.predict([img_array, img_array])[0][0]

        confidence_pneumonia = float(prediction)
        confidence_normal = 1.0 - confidence_pneumonia

        if confidence_pneumonia > 0.5:
            conf_display = confidence_pneumonia
            bar_width = int(conf_display * 100)
            st.markdown(f"""
            <div class="result-card pneumonia">
                <div class="result-icon">⚠️</div>
                <div>
                    <div class="result-label pneumonia">Pneumonia Detected</div>
                    <div class="result-conf">Model confidence: {conf_display:.1%}</div>
                </div>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-label">Confidence Score</div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill-pneumonia" style="width:{bar_width}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            conf_display = confidence_normal
            bar_width = int(conf_display * 100)
            st.markdown(f"""
            <div class="result-card normal">
                <div class="result-icon">✅</div>
                <div>
                    <div class="result-label normal">Normal</div>
                    <div class="result-conf">Model confidence: {conf_display:.1%}</div>
                </div>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-label">Confidence Score</div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill-normal" style="width:{bar_width}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Raw scores
        st.markdown(f"""
        <div style="margin-top:1.2rem; padding:1rem; background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06); border-radius:10px;">
            <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                        color:#475569; text-transform:uppercase; letter-spacing:0.08em;
                        margin-bottom:0.6rem;">Raw Scores</div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-family:'DM Mono',monospace; font-size:0.8rem; color:#f87171;">
                    Pneumonia&nbsp;&nbsp;{confidence_pneumonia:.4f}
                </span>
                <span style="font-family:'DM Mono',monospace; font-size:0.8rem; color:#4ade80;">
                    Normal&nbsp;&nbsp;{confidence_normal:.4f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-section" style="margin-top:2.5rem">
    <div class="footer-icon">🔬</div>
    <div class="footer-text">
        <strong>Research Disclaimer:</strong> This system is a proof-of-concept built
        during an ML Research Internship at Researcher's Connect. It is intended for
        academic demonstration only and is <strong>not validated for clinical use</strong>.
        Do not use for medical diagnosis. Always consult a licensed radiologist.
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown(""" 
<div style="text-align:center; margin-top:1.5rem; font-family:'DM Mono', monospace; font-size:0.75rem; color:#475569;">
    Built by <strong style="color:#38bdf8;">Sahil Sontakke</strong>
</div>
""", unsafe_allow_html=True)
