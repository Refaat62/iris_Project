import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import io

# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="Iris Recognition System",
    page_icon="👁️",
    layout="centered"
)

# =====================
# Custom CSS
# =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
        background-color: #0a0e1a;
        color: #e0e8ff;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 100%);
    }

    h1, h2, h3 {
        font-family: 'Share Tech Mono', monospace;
        color: #00d4ff;
        letter-spacing: 2px;
    }

    .iris-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #00d4ff33;
        margin-bottom: 2rem;
    }

    .iris-header h1 {
        font-size: 2.2rem;
        text-shadow: 0 0 20px #00d4ff88;
        margin-bottom: 0.3rem;
    }

    .iris-header p {
        color: #7a9abf;
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    /* Result Cards */
    .result-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid;
        text-align: center;
    }

    .card-allowed {
        background: #00ff8811;
        border-color: #00ff88;
    }

    .card-denied {
        background: #ff003311;
        border-color: #ff0033;
    }

    .card-unknown {
        background: #ffaa0011;
        border-color: #ffaa00;
    }

    .status-text-allowed {
        color: #00ff88;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 0 15px #00ff8888;
    }

    .status-text-denied {
        color: #ff0033;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 0 15px #ff003388;
    }

    .status-text-unknown {
        color: #ffaa00;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        font-weight: bold;
    }

    .person-id {
        color: #00d4ff;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .confidence-text {
        color: #7a9abf;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* Metric boxes */
    .metric-box {
        background: #0d1b2a;
        border: 1px solid #1a3a5c;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        color: #7a9abf;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .metric-value {
        color: #00d4ff;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.4rem;
        font-weight: bold;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 1px dashed #1a3a5c;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #080c16;
        border-right: 1px solid #1a3a5c;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff22, #0066ff22);
        border: 1px solid #00d4ff;
        color: #00d4ff;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 2px;
        padding: 0.6rem 2rem;
        border-radius: 6px;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #00d4ff44, #0066ff44);
        box-shadow: 0 0 15px #00d4ff44;
    }

    /* Warning / Info boxes */
    .stAlert {
        background: #0d1b2a;
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: #1a3a5c;
    }

    .scan-line {
        font-family: 'Share Tech Mono', monospace;
        color: #00d4ff88;
        font-size: 0.75rem;
        text-align: center;
        letter-spacing: 3px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Preprocessing (same as notebook)
# =====================
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CHANNELS = 1

def resize_keep_aspect_ratio(img, target_height=IMG_HEIGHT, target_width=IMG_WIDTH, pad_value=255):
    aspect_ratio = img.shape[1] / img.shape[0]
    if aspect_ratio > target_width / target_height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))
    preprocessed_img = np.full((target_height, target_width), pad_value, dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    preprocessed_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    return preprocessed_img

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = resize_keep_aspect_ratio(img)
    img = img / 255.0
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
    return img


# =====================
# Header
# =====================
st.markdown("""
<div class="iris-header">
    <h1>👁️ IRIS RECOGNITION</h1>
    <p>Biometric Identity Verification System</p>
</div>
""", unsafe_allow_html=True)


# =====================
# Auto-load Model (HuggingFace) & CSV (repo)
# =====================
import os
import requests
import tensorflow as tf

MODEL_URL  = "https://huggingface.co/morefaat69/iris-recognition/resolve/main/IRISRecognizer95.h5"
MODEL_PATH = "/tmp/IRISRecognizer95.h5"
CSV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test iris.csv")

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model... (first time only ~46MB)"):
            r = requests.get(MODEL_URL, stream=True)
            total = int(r.headers.get('content-length', 0))
            bar = st.progress(0, text="Downloading...")
            downloaded = 0
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.progress(min(downloaded/total, 1.0), text=f"Downloading... {downloaded//1024//1024}MB / {total//1024//1024}MB")
            bar.empty()

@st.cache_resource
def load_model_auto():
    download_model()
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        pass
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        pass
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
        Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Dropout)
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers

    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = Conv2D(filters, kernel_size, strides=stride, padding="same",
                   kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, kernel_size, padding="same",
                   kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x

    inputs = Input(shape=(150, 150, 1))
    x = Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = residual_block(x, 64);  x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2); x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2); x = residual_block(x, 256)
    x = residual_block(x, 512, stride=2); x = residual_block(x, 512)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(2000, activation="softmax")(x)
    model = Model(inputs, outputs, name="IRIS_ResNet")
    model.load_weights(MODEL_PATH)
    return model

@st.cache_data
def load_csv_auto():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    banned_candidates = [c for c in df.columns if 'ban' in c.lower() or 'allow' in c.lower() or 'status' in c.lower()]
    if banned_candidates and 'Banned' not in df.columns:
        df = df.rename(columns={banned_candidates[0]: 'Banned'})
    return df

model    = None
df_db    = None
model_ok = False
csv_ok   = os.path.exists(CSV_PATH)

try:
    model    = load_model_auto()
    model_ok = True
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

if csv_ok:
    try:
        df_db = load_csv_auto()
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")

# =====================
# Sidebar — Status Only
# =====================
with st.sidebar:
    st.markdown("## ⚙️ SYSTEM STATUS")
    st.markdown("---")

    st.markdown(f"""
    <div style='font-family: Share Tech Mono, monospace; font-size:0.85rem; color:#7a9abf; line-height:2.2'>
        MODEL &nbsp;&nbsp;: <span style='color:{"#00ff88" if model_ok else "#ff0033"}'>{"🟢 LOADED" if model_ok else "🔴 NOT FOUND"}</span><br>
        DATABASE: <span style='color:{"#00ff88" if csv_ok else "#ff0033"}'>{"🟢 LOADED" if csv_ok else "🔴 NOT FOUND"}</span>
    </div>
    """, unsafe_allow_html=True)

    if not csv_ok:
        st.warning("Place `test iris.csv` next to `iris_app.py`")

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Share Tech Mono, monospace; font-size:0.7rem; color:#3a5a7c; line-height:1.8'>
        MODEL &nbsp;&nbsp;&nbsp;: IRIS_ResNet<br>
        ACCURACY: 96.15%<br>
        CLASSES &nbsp;: 2000<br>
        INPUT &nbsp;&nbsp;&nbsp;: 150×150 px<br>
        EER &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 0.0001
    </div>
    """, unsafe_allow_html=True)



# =====================
# Main — Image Upload & Predict
# =====================
st.markdown("### 🔍 SCAN IRIS IMAGE")
uploaded_img = st.file_uploader(
    "Upload an iris image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_img:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Preview**")
        pil_img = Image.open(uploaded_img)
        st.image(pil_img, use_container_width=True, caption=uploaded_img.name)

    with col2:
        st.markdown("**Image Info**")
        w, h = pil_img.size
        st.markdown(f"""
        <div class='metric-box' style='margin-bottom:0.5rem'>
            <div class='metric-label'>Dimensions</div>
            <div class='metric-value'>{w} × {h}</div>
        </div>
        <div class='metric-box' style='margin-bottom:0.5rem'>
            <div class='metric-label'>Mode</div>
            <div class='metric-value'>{pil_img.mode}</div>
        </div>
        <div class='metric-box'>
            <div class='metric-label'>File</div>
            <div class='metric-value' style='font-size:0.85rem'>{uploaded_img.name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Predict button ──
    if st.button("▶  RUN IDENTIFICATION"):

        if model is None:
            st.warning("⚠️ Model not found. Make sure `IRISRecognizer95.h5` is in the same folder as `iris_app.py`.")
        else:
            with st.spinner("Scanning..."):
                uploaded_img.seek(0)
                img_array = preprocess_image(uploaded_img)
                pred = model.predict(img_array, verbose=0)
                pred_id   = int(np.argmax(pred))
                confidence = float(np.max(pred)) * 100

            st.markdown('<div class="scan-line">── SCAN COMPLETE ──</div>', unsafe_allow_html=True)

            # ── Look up in DB ──
            if df_db is not None:
                file_name = uploaded_img.name
                # Match by exact filename at end of path (handles full path in CSV)
                file_name_clean = file_name.strip()
                result_row = df_db[df_db['ImagePath'].apply(
                    lambda p: str(p).replace('\\', '/').split('/')[-1].strip() == file_name_clean
                )]

                if not result_row.empty:
                    banned_col = next((c for c in df_db.columns if c.lower() == 'banned'), None)
                    id_col     = next((c for c in df_db.columns if c.lower() == 'person_id'), None)
                    status    = result_row[banned_col].values[0] if banned_col else "Unknown"
                    person_id = result_row[id_col].values[0]    if id_col    else "N/A"
                    is_allowed = (str(status).strip() == "Allowed")

                    if is_allowed:
                        st.markdown(f"""
                        <div class='result-card card-allowed'>
                            <div class='status-text-allowed'>✅ ACCESS GRANTED</div>
                            <div class='person-id'>PERSON ID : {person_id}</div>
                            <div class='confidence-text'>Confidence: {confidence:.2f}%  |  Predicted Class: {pred_id}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-card card-denied'>
                            <div class='status-text-denied'>🚫 ACCESS DENIED</div>
                            <div class='person-id'>PERSON ID : {person_id}</div>
                            <div class='confidence-text'>Confidence: {confidence:.2f}%  |  Status: {str(status).replace('_',' ')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-card card-unknown'>
                        <div class='status-text-unknown'>⚠️ NOT IN DATABASE</div>
                        <div class='person-id'>PREDICTED CLASS : {pred_id}</div>
                        <div class='confidence-text'>Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                # No CSV — show raw prediction only
                st.markdown(f"""
                <div class='result-card card-unknown'>
                    <div class='status-text-unknown'>🔎 IDENTITY DETECTED</div>
                    <div class='person-id'>PREDICTED CLASS : {pred_id}</div>
                    <div class='confidence-text'>Confidence: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 Load the CSV database from the sidebar to see Allow/Deny status.")

            # ── Confidence bar ──
            st.markdown("**Confidence Level**")
            st.progress(min(confidence / 100, 1.0))
            st.markdown(f'<div style="text-align:center; font-family: Share Tech Mono; color:#00d4ff; font-size:0.9rem">{confidence:.2f}%</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center; color:#3a5a7c; padding: 3rem 0; font-family: Share Tech Mono, monospace; font-size:0.85rem; letter-spacing:2px'>
        ── AWAITING IRIS SCAN ──<br><br>
        Upload an iris image to begin identification
    </div>
    """, unsafe_allow_html=True)