import cv2
import numpy as np
import mediapipe as mp
import os
import csv
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array

# --- 1. SENSOR INIT ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
LOG_FILE = "biometric_data.csv"

# --- 2. DATA ARCHITECTURE ---
def init_biometric_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Emotion", "Confidence", "Smile", "Brow", "Squint"])

def log_to_csv(emotion, confidence, au_data):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, emotion, f"{confidence:.2f}", 
                         f"{au_data['Smile']:.2f}", f"{au_data['Brow']:.2f}", f"{au_data['Squint']:.2f}"])

# --- 3. VISION PIPELINE ---
def get_face_mesh_results(frame):
    return face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def calculate_au_extended(landmarks, w, h):
    c = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
    smile = np.linalg.norm(np.array(c[61]) - np.array(c[291])) / (np.linalg.norm(np.array(c[0]) - np.array(c[17])) + 1e-6)
    brow = np.clip(1.0 - (np.linalg.norm(np.array(c[21]) - np.array(c[251])) / 65.0), 0, 1)
    squint = np.clip(1.0 - (np.linalg.norm(np.array(c[159]) - np.array(c[145])) / 18.0), 0, 1)
    return {'Smile': smile, 'Brow': brow, 'Squint': squint}

def preprocess_face(frame, bbox):
    x, y, w, h = bbox
    roi = frame[max(0,y):min(frame.shape[0],y+h), max(0,x):min(frame.shape[1],x+w)]
    if roi.size == 0: return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    return np.expand_dims(img_to_array(resized), axis=0) / 255.0

def draw_clean_hud(frame, bbox, emotion, conf):
    x, y, w, h = bbox
    cyan = (212, 212, 38) # Matches #26D4D4
    cv2.rectangle(frame, (x, y), (x+w, y+h), cyan, 1)
    cv2.putText(frame, f"{emotion} {int(conf*100)}%", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cyan, 1)