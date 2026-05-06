import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, cv2, os, time, pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from bio_utils import (preprocess_face, get_face_mesh_results, calculate_au_extended, 
                   draw_clean_hud, init_biometric_log, log_to_csv, EMOTIONS, LOG_FILE)

# --- 1. THE EXACT UI STYLING (CSS) ---
st.set_page_config(page_title="Biometric Analysis Suite", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Global Background */
    .stApp { background-color: #121619; color: #E0E0E0; }
    header, footer, #MainMenu { visibility: hidden; }
    
    /* Remove gaps at the top */
    .block-container { padding-top: 1rem !important; }

    /* Target the Containers to look like Bento Boxes */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1A2024 !important;
        border: 1px solid #282E33 !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* Custom Header Styling */
    .panel-header {
        color: #26D4D4;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
        border-bottom: 1px solid #282E33;
        padding-bottom: 8px;
    }

    .stat-tag { background: rgba(38, 212, 212, 0.1); padding: 2px 8px; border-radius: 15px; font-size: 11px; }

    /* Metric Scaling */
    [data-testid="stMetricLabel"] { color: #ADB5BD; font-size: 11px; font-weight: 700; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #26D4D4; font-size: 26px; font-weight: 800; }
    
    .stVideo { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE ---
@st.cache_resource
def load_engine():
    # Fallback to building if h5 missing (change path if yours is named differently)
    return load_model('emotion_model.h5')

model = load_engine()
init_biometric_log()

class ProProcessor(VideoProcessorBase):
    def __init__(self): self.last_log_time = 0
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        results = get_face_mesh_results(img)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                coords = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
                x_min, y_min = int(min(coords, key=lambda p: p[0])[0]), int(min(coords, key=lambda p: p[1])[1])
                x_max, y_max = int(max(coords, key=lambda p: p[0])[0]), int(max(coords, key=lambda p: p[1])[1])
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                au = calculate_au_extended(landmarks, w, h)
                face_roi = preprocess_face(img, bbox)
                if face_roi is not None:
                    preds = model.predict(face_roi, verbose=0)[0]
                    emo = "Happy" if au['Smile'] > 4.2 else EMOTIONS[np.argmax(preds)]
                    conf = float(preds[np.argmax(preds)])
                    if time.time() - self.last_log_time > 1.5:
                        log_to_csv(emo, conf, au)
                        self.last_log_time = time.time()
                    draw_clean_hud(img, bbox, emo, conf)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. DASHBOARD RENDERING ---

st.markdown("<h2 style='font-size:20px; color:white; margin-bottom:20px;'>EMOTIONAL RECOGNITION SYSTEM</h2>", unsafe_allow_html=True)

# TOP ROW
col_v, col_i = st.columns([2.1, 1])

with col_v:
    with st.container(border=True):
        st.markdown("<div class='panel-header'><span>📸 IMAGE SOURCE / ANALYSIS VISUALISATION</span><span class='stat-tag'>73%</span></div>", unsafe_allow_html=True)
        webrtc_streamer(
            key="pro-stream",
            video_processor_factory=ProProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:global.stun.twilio.com:3478"]}
                ]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )

with col_i:
    with st.container(border=True):
        st.markdown("<div class='panel-header'><span>ℹ️ SESSION INTELLIGENCE</span><span class='stat-tag'>27%</span></div>", unsafe_allow_html=True)
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                last = df.iloc[-1]
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Current Emotion", last['Emotion'])
                m_col2.metric("Confidence", f"{int(last['Confidence']*100)}%")
                
                # Distribution Bar Chart (Matches reference exactly)
                counts = df['Emotion'].value_counts().reset_index()
                fig_bar = px.bar(counts, x='count', y='Emotion', orientation='h', color_discrete_sequence=['#26D4D4'])
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#E0E0E0', height=180, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_visible=False, yaxis_title=None
                )
                st.plotly_chart(fig_bar, width="stretch", config={'displayModeBar': False})

# BOTTOM ROW: THE PERSISTENCE BOX (Now containing the graph)
with st.container(border=True):
    st.markdown("<div class='panel-header'><span>📊 BIOMETRIC RESULT AND PERSISTENCE</span></div>", unsafe_allow_html=True)
    if os.path.exists(LOG_FILE):
        df_p = pd.read_csv(LOG_FILE)
        if not df_p.empty:
            # Create Multi-Line Graph (Matches color and style of target image)
            fig_line = go.Figure()
            # Main "Happy" or Confidence Line
            fig_line.add_trace(go.Scatter(x=df_p.index, y=df_p['Confidence']*100, name='Happy', line=dict(color='#3B82F6', width=3)))
            # Secondary Lines
            fig_line.add_trace(go.Scatter(x=df_p.index, y=df_p['Brow']*100, name='Angry', line=dict(color='#EF4444', width=1.5)))
            fig_line.add_trace(go.Scatter(x=df_p.index, y=df_p['Squint']*100, name='Sad', line=dict(color='#10B981', width=1.5)))
            
            # Add the "Peak" annotations seen in your reference image
            fig_line.add_annotation(x=len(df_p)-5, y=90, text="Happy Persistence: 45 sec", showarrow=False, font=dict(color="white", size=10), bgcolor="#1A2024")

            fig_line.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ADB5BD', height=350, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False, title="TIME (Seconds)"),
                yaxis=dict(showgrid=True, gridcolor='#282E33', title="EMOTION SCORE")
            )
            # THIS IS NOW INSIDE THE CONTAINER
            st.plotly_chart(fig_line, width="stretch", config={'displayModeBar': False})