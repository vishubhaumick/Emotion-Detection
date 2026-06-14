import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EMOTION SCAN v1.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_emotion_model():
    model = load_model('emotion_final_model.h5')
    with open('emotion_labels.json', 'r') as f:
        labels = json.load(f)
    return model, labels

model, emotion_labels = load_emotion_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (96, 96))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    return np.expand_dims(face_normalized, axis=0)

def predict_emotion(face_input):
    prediction = model.predict(face_input, verbose=0)[0]
    emotion_idx = np.argmax(prediction)
    return emotion_labels[emotion_idx], prediction[emotion_idx] * 100, prediction

def draw_results(image, faces, emotions):
    result_img = image.copy()
    for (x, y, w, h), (emotion, confidence, _) in zip(faces, emotions):
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (245, 216, 0), 2)
        label = f"{emotion.upper()} {confidence:.1f}%"
        cv2.putText(result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 216, 0), 2)
    return result_img

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>
:root {
  --cp-black: #080810;
  --cp-panel: #0D0D1A;
  --cp-glass: rgba(13,13,26,0.72);
  --cp-yellow: #F5D800;
  --cp-cyan: #00F0FF;
  --cp-red: #FF003C;
  --cp-border: rgba(245,216,0,0.35);
  --cp-text: #E4E4F0;
  --cp-muted: #6A6A8A;
  --cp-glow-y: 0 0 8px #F5D800, 0 0 20px rgba(245,216,0,0.4);
  --cp-glow-c: 0 0 8px #00F0FF, 0 0 20px rgba(0,240,255,0.4);
  --font-display: 'Rajdhani', sans-serif;
  --font-mono: 'Share Tech Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--font-display) !important;
  background-color: var(--cp-black) !important;
  color: var(--cp-text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cp-panel); }
::-webkit-scrollbar-thumb { background: var(--cp-yellow); border-radius: 3px; }

body::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 9999;
  pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.07) 2px,
    rgba(0,0,0,0.07) 4px
  );
}

[data-testid="stSidebar"] {
  background: var(--cp-panel) !important;
  border-right: 1px solid var(--cp-border) !important;
}
[data-testid="stSidebar"] * { color: var(--cp-text) !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

.main .block-container {
  padding-top: 2rem !important;
  max-width: 1400px !important;
}

[data-testid="stTabs"] button {
  font-family: var(--font-mono) !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.08em !important;
  color: var(--cp-muted) !important;
  border: none !important;
  background: transparent !important;
  padding: 0.5rem 1.2rem !important;
  transition: color 0.2s, border-color 0.2s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--cp-yellow) !important;
  border-bottom: 2px solid var(--cp-yellow) !important;
  text-shadow: var(--cp-glow-y) !important;
}
[data-testid="stTabs"] button:hover { color: var(--cp-yellow) !important; }
[data-testid="stTabsTabPanel"] { padding-top: 1.5rem !important; }

[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] span[data-testid="stCheckboxLabel"] {
  font-family: var(--font-mono) !important;
  color: var(--cp-cyan) !important;
  letter-spacing: 0.05em;
}

[data-testid="stFileUploader"] {
  border: 1px dashed var(--cp-border) !important;
  border-radius: 4px !important;
  background: rgba(245,216,0,0.03) !important;
  padding: 0.5rem;
}
[data-testid="stFileUploader"] button {
  background: transparent !important;
  border: 1px solid var(--cp-yellow) !important;
  color: var(--cp-yellow) !important;
  font-family: var(--font-mono) !important;
  border-radius: 3px !important;
}

[data-testid="stAlert"] {
  background: rgba(255,0,60,0.1) !important;
  border: 1px solid var(--cp-red) !important;
  border-radius: 4px !important;
  font-family: var(--font-mono) !important;
}

[data-testid="stImage"] img {
  border: 1px solid var(--cp-border) !important;
  border-radius: 4px !important;
}

@keyframes glitch {
  0% { clip-path: inset(0 0 98% 0); transform: translate(-2px, 0); }
  10% { clip-path: inset(40% 0 50% 0); transform: translate(2px, 0); }
  20% { clip-path: inset(80% 0 10% 0); transform: translate(-1px, 0); }
  30% { clip-path: inset(10% 0 80% 0); transform: translate(1px, 0); }
  40% { clip-path: inset(60% 0 30% 0); transform: translate(-2px, 0); }
  50% { clip-path: inset(20% 0 70% 0); transform: translate(0, 0); }
  60% { clip-path: inset(90% 0 5% 0); transform: translate(2px, 0); }
  70% { clip-path: inset(5% 0 90% 0); transform: translate(-1px, 0); }
  80% { clip-path: inset(50% 0 40% 0); transform: translate(1px, 0); }
  90% { clip-path: inset(30% 0 60% 0); transform: translate(-2px, 0); }
  100% { clip-path: inset(0 0 98% 0); transform: translate(0, 0); }
}

@keyframes scanmove {
  0% { top: -10%; }
  100% { top: 110%; }
}

@keyframes flicker {
  0%, 95%, 100% { opacity: 1; }
  96% { opacity: 0.85; }
  97% { opacity: 1; }
  98% { opacity: 0.9; }
  99% { opacity: 1; }
}

@keyframes bootline {
  from { opacity: 0; transform: translateX(-8px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes pulse-border {
  0%, 100% { box-shadow: 0 0 10px rgba(245,216,0,0.3), inset 0 0 10px rgba(245,216,0,0.05); }
  50% { box-shadow: 0 0 25px rgba(245,216,0,0.6), inset 0 0 20px rgba(245,216,0,0.1); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

#cp-welcome-overlay {
  position: fixed;
  inset: 0;
  z-index: 99999;
  background: rgba(5,5,15,0.97);
  display: flex;
  align-items: center;
  justify-content: center;
  animation: flicker 6s infinite;
}

.cp-welcome-panel {
  position: relative;
  width: min(640px, 92vw);
  background: var(--cp-glass);
  border: 1px solid var(--cp-yellow);
  border-radius: 4px;
  padding: 3rem 3.5rem;
  box-shadow: var(--cp-glow-y);
  backdrop-filter: blur(24px);
  animation: pulse-border 3s ease-in-out infinite, fadeUp 0.6s cubic-bezier(0.16,1,0.3,1) both;
  overflow: hidden;
}

.cp-welcome-panel::before {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, var(--cp-cyan), transparent);
  animation: scanmove 3s linear infinite;
  opacity: 0.6;
}

.cp-welcome-panel::after {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(135deg, rgba(245,216,0,0.06) 0%, transparent 40%),
    linear-gradient(315deg, rgba(0,240,255,0.04) 0%, transparent 40%);
  pointer-events: none;
  border-radius: 4px;
}

.cp-corner { position: absolute; width: 16px; height: 16px; }
.cp-corner-tl { top: -1px; left: -1px; border-top: 2px solid var(--cp-cyan); border-left: 2px solid var(--cp-cyan); }
.cp-corner-tr { top: -1px; right: -1px; border-top: 2px solid var(--cp-cyan); border-right: 2px solid var(--cp-cyan); }
.cp-corner-bl { bottom: -1px; left: -1px; border-bottom: 2px solid var(--cp-cyan); border-left: 2px solid var(--cp-cyan); }
.cp-corner-br { bottom: -1px; right: -1px; border-bottom: 2px solid var(--cp-cyan); border-right: 2px solid var(--cp-cyan); }

.cp-welcome-eyecon {
  text-align: center;
  font-size: 3rem;
  margin-bottom: 0.5rem;
  filter: drop-shadow(0 0 12px var(--cp-cyan));
  animation: flicker 4s infinite;
}

.cp-welcome-pretitle {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.3em;
  color: var(--cp-cyan);
  text-align: center;
  margin-bottom: 0.25rem;
  text-shadow: var(--cp-glow-c);
}

.cp-welcome-title {
  font-family: var(--font-display);
  font-size: clamp(2rem, 6vw, 3rem);
  font-weight: 700;
  text-align: center;
  color: var(--cp-yellow);
  text-shadow: var(--cp-glow-y);
  letter-spacing: 0.06em;
  line-height: 1.1;
  margin-bottom: 0.25rem;
  position: relative;
}

.cp-welcome-title::after {
  content: attr(data-text);
  position: absolute;
  inset: 0;
  color: var(--cp-cyan);
  animation: glitch 5s steps(1) infinite;
  opacity: 0.7;
}

.cp-welcome-version,
.cp-page-subtitle,
.cp-stat-label,
.cp-info-box,
.cp-confidence {
  font-family: var(--font-mono);
}

.cp-welcome-version {
  font-size: 0.75rem;
  color: var(--cp-muted);
  text-align: center;
  letter-spacing: 0.15em;
  margin-bottom: 1.75rem;
}

.cp-boot-lines {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  line-height: 1.9;
  margin-bottom: 2rem;
  border-left: 2px solid var(--cp-border);
  padding-left: 1rem;
}
.cp-boot-line { opacity: 0; animation: bootline 0.3s ease forwards; }
.cp-boot-line:nth-child(1) { animation-delay: 0.3s; }
.cp-boot-line:nth-child(2) { animation-delay: 0.7s; }
.cp-boot-line:nth-child(3) { animation-delay: 1.1s; }
.cp-boot-line:nth-child(4) { animation-delay: 1.5s; }
.cp-boot-line:nth-child(5) { animation-delay: 1.9s; }
.cp-boot-line:nth-child(6) { animation-delay: 2.3s; }

.cp-ok { color: var(--cp-cyan); }
.cp-warn { color: var(--cp-yellow); }

.cp-tagline {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--cp-muted);
  text-align: center;
  letter-spacing: 0.05em;
  margin-bottom: 2.25rem;
  font-style: italic;
}

.cp-page-header {
  border-bottom: 1px solid var(--cp-border);
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}

.cp-page-pretitle,
.cp-sidebar-title,
.cp-section-head {
  font-family: var(--font-mono);
  color: var(--cp-cyan);
  text-shadow: var(--cp-glow-c);
}

.cp-page-pretitle {
  font-size: 0.7rem;
  letter-spacing: 0.3em;
}

.cp-page-title {
  font-family: var(--font-display);
  font-size: clamp(1.8rem, 4vw, 2.8rem);
  font-weight: 700;
  color: var(--cp-yellow);
  text-shadow: var(--cp-glow-y);
  letter-spacing: 0.05em;
  line-height: 1.1;
  margin: 0.15rem 0 0.4rem;
}

.cp-page-subtitle {
  font-size: 0.78rem;
  color: var(--cp-muted);
  letter-spacing: 0.08em;
}

.cp-sidebar-title {
  font-size: 0.65rem;
  letter-spacing: 0.3em;
  border-bottom: 1px solid var(--cp-border);
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
}

.cp-stat-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.4rem 0;
  border-bottom: 1px solid rgba(245,216,0,0.07);
}

.cp-stat-value {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--cp-yellow);
  text-shadow: 0 0 6px rgba(245,216,0,0.5);
}

.cp-emotion-tag {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.08em;
  padding: 0.2rem 0.5rem;
  border: 1px solid var(--cp-border);
  border-radius: 2px;
  color: var(--cp-text);
  margin: 0.2rem 0.15rem;
  background: rgba(245,216,0,0.05);
}

.cp-section-head {
  font-size: 0.68rem;
  letter-spacing: 0.25em;
  border-left: 3px solid var(--cp-cyan);
  padding-left: 0.6rem;
  margin-bottom: 0.75rem;
}

.cp-result-panel {
  border: 1px solid var(--cp-border);
  border-radius: 4px;
  padding: 1.25rem;
  background: rgba(245,216,0,0.025);
  position: relative;
}

.cp-result-panel::before {
  content: "// OUTPUT";
  position: absolute;
  top: -0.55rem;
  left: 1rem;
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  color: var(--cp-yellow);
  background: var(--cp-black);
  padding: 0 0.4rem;
}

.cp-emotion-result {
  font-family: var(--font-display);
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: var(--cp-yellow);
  text-shadow: var(--cp-glow-y);
}

.cp-webcam-frame {
  border: 1px solid var(--cp-border);
  border-radius: 4px;
  padding: 0.5rem;
  background: var(--cp-panel);
  position: relative;
}

.cp-webcam-frame::before {
  content: "● LIVE";
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  color: var(--cp-red);
  text-shadow: 0 0 8px var(--cp-red);
  position: absolute;
  top: -0.55rem;
  left: 1rem;
  background: var(--cp-black);
  padding: 0 0.4rem;
}

.cp-info-box {
  font-size: 0.75rem;
  color: var(--cp-muted);
  border: 1px solid rgba(0,240,255,0.15);
  border-radius: 3px;
  padding: 0.65rem 1rem;
  background: rgba(0,240,255,0.03);
  line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

if "welcome_done" not in st.session_state:
    st.session_state.welcome_done = False

if not st.session_state.welcome_done:
    st.markdown("""
    <style>
    .cp-welcome-wrap {
        position: relative;
        margin: 2rem auto 0 auto;
        max-width: 760px;
        border: 1px solid rgba(245,216,0,0.45);
        border-radius: 6px;
        padding: 2rem;
        background: rgba(13,13,26,0.86);
        box-shadow: 0 0 24px rgba(245,216,0,0.22);
        backdrop-filter: blur(18px);
    }
    .cp-welcome-kicker {
        font-family: 'Share Tech Mono', monospace;
        color: #00F0FF;
        letter-spacing: 0.28em;
        font-size: 0.72rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .cp-welcome-head {
        font-family: 'Rajdhani', sans-serif;
        color: #F5D800;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
        text-shadow: 0 0 10px rgba(245,216,0,0.45);
    }
    .cp-welcome-sub {
        font-family: 'Share Tech Mono', monospace;
        color: #6A6A8A;
        text-align: center;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        margin-bottom: 1.4rem;
    }
    .cp-welcome-console {
        font-family: 'Share Tech Mono', monospace;
        color: #cfd3ff;
        font-size: 0.82rem;
        line-height: 1.9;
        border-left: 2px solid rgba(245,216,0,0.3);
        padding-left: 1rem;
        margin-bottom: 1.2rem;
        white-space: pre-line;
    }
    .cp-welcome-tagline {
        text-align: center;
        color: #8E90B5;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cp-welcome-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="cp-welcome-kicker">NEURAL INTERFACE SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="cp-welcome-head">EMOTION SCAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="cp-welcome-sub">v1.0 / MOBILENETV2 / FER2013 4X</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="cp-welcome-console">
        &gt; INITIALIZING NEURAL NETWORK...<br>
        &gt; MODEL: MobileNetV2 [ LOADED ]<br>
        &gt; DATASET: FER2013 4X [ READY ]<br>
        &gt; ACCURACY: 63.34% [ CALIBRATED ]<br>
        &gt; EMOTION CLASSES: 7 [ UNLOCKED ]<br>
        &gt; AWAITING USER INPUT_
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="cp-welcome-tagline">Reading the human face, one neuron at a time.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c2:
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        if st.button("ENTER SYSTEM", type="primary"):
            st.session_state.welcome_done = True
            st.experimental_rerun()

    st.stop()

with st.sidebar:
    st.markdown("""
    <div class="cp-sidebar-title">◈ SYSTEM INFO</div>
    <div class="cp-stat-row"><span class="cp-stat-label">MODEL</span><span class="cp-stat-value">MobileNetV2</span></div>
    <div class="cp-stat-row"><span class="cp-stat-label">DATASET</span><span class="cp-stat-value">FER2013 4X</span></div>
    <div class="cp-stat-row"><span class="cp-stat-label">ACCURACY</span><span class="cp-stat-value">63.34%</span></div>
    <div class="cp-stat-row"><span class="cp-stat-label">CLASSES</span><span class="cp-stat-value">7</span></div>
    <br>
    <div class="cp-sidebar-title">◈ EMOTION CLASSES</div>
    """, unsafe_allow_html=True)

    tags_html = "".join(f'<span class="cp-emotion-tag">{lbl.upper()}</span>' for lbl in emotion_labels)
    st.markdown(f'<div style="margin-bottom:1.5rem">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cp-sidebar-title" style="margin-top:1rem">◈ STATUS</div>
    <div class="cp-info-box">
      FACE DETECTOR : HAAR CASCADE<br>
      INPUT SIZE : 96 × 96 px<br>
      FRAMEWORK : TF / Keras<br>
      INTERFACE : Streamlit
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="cp-page-header">
  <div class="cp-page-pretitle">// NEURAL INTERFACE · ACTIVE</div>
  <div class="cp-page-title">EMOTION SCAN</div>
  <div class="cp-page-subtitle">HUMAN EMOTIONAL STATE ANALYSIS · REAL-TIME DETECTION SYSTEM</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔴 LIVE FEED", "📁 UPLOAD IMAGE"])

with tab1:
    st.markdown('<div class="cp-section-head">REAL-TIME SENSOR (CLOUD BRIDGE)</div>', unsafe_allow_html=True)
    
    # Streamlit's native client-side camera widget (works on phones, Macs, Windows browsers)
    camera_image = st.camera_input("Initialize Secure Optical Link")

    if camera_image is not None:
        # Convert the browser's image file into an OpenCV readable matrix
        image = Image.open(camera_image)
        frame = np.array(image)
        
        # st.camera_input returns RGB, but OpenCV Haar Cascades expect grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            st.markdown("""
            <div class="cp-info-box" style="border-color:rgba(255,0,60,0.4);background:rgba(255,0,60,0.04)">
              ⚠ NO FACIAL GEOMETRY DETECTED. PLEASE ADJUST LIGHTING.
            </div>
            """, unsafe_allow_html=True)
        else:
            all_emotions = []
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
                    
                face_input = preprocess_face(face_crop)
                emotion, confidence, prediction = predict_emotion(face_input)
                all_emotions.append((emotion, confidence, prediction))

            # Re-use your existing draw_results function for UI consistency
            result_image = draw_results(frame, faces, all_emotions)
            
            st.markdown('<div class="cp-result-panel" style="margin-top:1rem">', unsafe_allow_html=True)
            st.image(result_image, caption="BIOMETRIC DECRYPTION COMPLETE", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="cp-section-head">STATIC IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD TARGET IMAGE", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            st.markdown("""
            <div class="cp-info-box" style="border-color:rgba(255,0,60,0.4);background:rgba(255,0,60,0.04)">
              ⚠ NO FACE DETECTED — UPLOAD A CLEAR FRONT-FACING IMAGE
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="cp-result-panel" style="margin-top:1rem">', unsafe_allow_html=True)
            st.image(image_rgb, caption="UPLOADED TARGET", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            all_emotions = []
            for (x, y, w, h) in faces:
                face_crop = image[y:y+h, x:x+w]
                face_input = preprocess_face(face_crop)
                emotion, confidence, prediction = predict_emotion(face_input)
                all_emotions.append((emotion, confidence, prediction))

            result_image = draw_results(image_rgb, faces, all_emotions)
            col1, col2 = st.columns([1.1, 1], gap="large")

            with col1:
                st.markdown('<div class="cp-section-head">SCAN RESULT</div>', unsafe_allow_html=True)
                st.markdown('<div class="cp-result-panel">', unsafe_allow_html=True)
                st.image(result_image, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="cp-section-head">EMOTION ANALYSIS</div>', unsafe_allow_html=True)

                for i, (emotion, confidence, prediction) in enumerate(all_emotions):
                    st.markdown(f'''
                    <div class="cp-result-panel" style="margin-bottom:1rem">
                      <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--cp-muted);letter-spacing:0.15em;margin-bottom:0.5rem">
                        TARGET FACE {i+1:02d}
                      </div>
                      <div class="cp-emotion-result">{emotion.upper()}</div>
                      <div class="cp-confidence">CONFIDENCE · {confidence:.2f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    fig.patch.set_facecolor('#0A0A10')
                    ax.set_facecolor('#0D0D1A')

                    bar_colors = [
                        '#F5D800' if j == np.argmax(prediction) else '#1E1E32'
                        for j in range(len(emotion_labels))
                    ]
                    edge_colors = [
                        '#F5D800' if j == np.argmax(prediction) else '#2A2A45'
                        for j in range(len(emotion_labels))
                    ]

                    bars = ax.barh(
                        [lbl.upper() for lbl in emotion_labels],
                        prediction * 100,
                        color=bar_colors,
                        edgecolor=edge_colors,
                        linewidth=0.8,
                        height=0.55
                    )

                    for bar, val in zip(bars, prediction * 100):
                        if val > 2:
                            ax.text(
                                val + 0.5,
                                bar.get_y() + bar.get_height() / 2,
                                f'{val:.1f}%',
                                va='center',
                                ha='left',
                                fontsize=6.5,
                                color='#6A6A8A',
                                fontfamily='monospace'
                            )

                    ax.set_xlabel('CONFIDENCE %', fontsize=7, color='#6A6A8A', labelpad=6)
                    ax.set_title(
                        f'FACE {i+1:02d} · PROBABILITY MATRIX',
                        fontsize=7.5,
                        color='#00F0FF',
                        pad=10
                    )
                    ax.set_xlim(0, 100)
                    ax.tick_params(axis='both', colors='#6A6A8A', labelsize=7)
                    ax.spines['bottom'].set_color('#2A2A45')
                    ax.spines['left'].set_color('#2A2A45')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    plt.tight_layout(pad=1.2)
                    st.pyplot(fig)
                    plt.close()

    else:
        st.markdown("""
        <div class="cp-info-box" style="margin-top:0.5rem">
          &gt; NO TARGET LOADED · UPLOAD A JPG / PNG IMAGE TO BEGIN ANALYSIS<br>
          &gt; SYSTEM WILL AUTO-DETECT ALL FACES IN THE IMAGE
        </div>
        """, unsafe_allow_html=True)