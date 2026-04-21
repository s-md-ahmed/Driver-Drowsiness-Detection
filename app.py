import streamlit as st
import cv2
import numpy as np
import base64
import os
import time
import pandas as pd
from datetime import datetime
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. IMPORT LOCAL MODULES ---
try:
    from model_loader import DrowsinessModelLoader
    from utils import calculate_ear, calculate_mar, estimate_head_pitch
    from detection import DrowsinessDetector
except ImportError as e:
    st.error(f"Missing local file: {e}")
    st.stop()

# --- 2. STREAMLIT UI CONFIG ---
st.set_page_config(page_title="Driver Monitor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
        text-align: center; border: 1px solid #333;
    }
    .metric-title { font-size: 14px; color: #888; text-transform: uppercase; }
    .metric-value { font-size: 28px; font-weight: bold; color: #fff; }
    .alert-box { background-color: #ff4b4b; color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    .awake-box { background-color: #4caf50; color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("🚙 Driver Drowsiness Detection")

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_models():
    model_path_mp = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path_mp)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options)
    
    model_path_pt = "drowsiness_detection_model_v1.pth"
    dl_model = DrowsinessModelLoader(model_path=model_path_pt, device='cpu') if os.path.exists(model_path_pt) else None
    return face_mesh, dl_model

face_mesh_model, pytorch_model = load_models()

LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
MOUTH_IDXS = [78, 81, 13, 311, 308, 402, 14, 178] 

# --- SESSION STATE ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False
if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0

# UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Feed")
    frame_placeholder = st.empty()
    if not st.session_state.is_running:
        if st.button("🚀 Start Camera", use_container_width=True, type="primary"):
            st.session_state.is_running = True
            st.rerun()
    else:
        if st.button("🛑 Stop Camera", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()

with col2:
    st.subheader("Live Metrics")
    alert_placeholder = st.empty()
    ear_placeholder = st.empty()
    mar_placeholder = st.empty()
    fatigue_placeholder = st.empty()
    audio_placeholder = st.empty()

detector = DrowsinessDetector(ear_threshold=0.25, mar_threshold=0.6, pitch_threshold=0.6, closed_time_threshold=2.0)

def play_alarm():
    try:
        if os.path.exists("alarm.wav"):
            audio_file = open("alarm.wav", "rb").read()
            b64_audio = base64.b64encode(audio_file).decode()
            unique_id = time.time() 
            audio_html = f"""<audio autoplay="true" key="{unique_id}"><source src="data:audio/wav;base64,{b64_audio}" type="audio/wav"></audio>"""
            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
    except:
        pass

# --- 4. MAIN LOOP ---
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    try:
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = face_mesh_model.detect(mp_image)
            
            current_ear, current_mar, current_pitch, model_eye_state = 0.0, 0.0, 1.0, 1
            
            if detection_result.face_landmarks:
                face_lms = detection_result.face_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in face_lms]
                
                current_ear = (calculate_ear([landmarks[i] for i in LEFT_EYE_IDXS]) + 
                               calculate_ear([landmarks[i] for i in RIGHT_EYE_IDXS])) / 2.0
                current_mar = calculate_mar([landmarks[i] for i in MOUTH_IDXS])
                current_pitch = estimate_head_pitch(landmarks, w, h)
                
                # Background DL Inference
                if pytorch_model:
                    try:
                        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                        simulated_ir = cv2.equalizeHist(gray) 
                        model_eye_state = pytorch_model.predict(simulated_ir) if hasattr(pytorch_model, 'predict') else 1
                    except:
                        model_eye_state = 1 

                alerts = detector.evaluate(current_ear, current_mar, current_pitch, model_eye_state)
                
                if alerts:
                    alert_placeholder.markdown(f'<div class="alert-box">🚨 {" | ".join(alerts)}</div>', unsafe_allow_html=True)
                    if [a for a in alerts if a != "HIGH FATIGUE SCORE"]:
                        now = time.time()
                        if not st.session_state.alarm_active or (now - st.session_state.last_alarm_time > 3.0):
                            play_alarm()
                            st.session_state.alarm_active = True
                            st.session_state.last_alarm_time = now
                else:
                    alert_placeholder.markdown('<div class="awake-box">✅ AWAKE</div>', unsafe_allow_html=True)
                    st.session_state.alarm_active = False
                    audio_placeholder.empty()
                
                # Visuals
                dot_color = (0, 255, 0) if current_ear > 0.25 else (255, 0, 0)
                for i in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
                    cx, cy = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
                    cv2.circle(rgb_frame, (cx, cy), 2, dot_color, -1)

            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            ear_placeholder.markdown(f'<div class="metric-card"><div class="metric-title">EAR</div><div class="metric-value">{current_ear:.2f}</div></div>', unsafe_allow_html=True)
            mar_placeholder.markdown(f'<div class="metric-card"><div class="metric-title">MAR</div><div class="metric-value">{current_mar:.2f}</div></div>', unsafe_allow_html=True)
            fatigue_placeholder.markdown(f'<div class="metric-card"><div class="metric-title">Fatigue</div><div class="metric-value">{detector.fatigue_score:.1f}</div></div>', unsafe_allow_html=True)
            
            time.sleep(0.01)
    finally:
        cap.release()
else:
    frame_placeholder.info("Ready for monitoring. Click Start.")
