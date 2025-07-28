import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pygame

# Streamlit configuration
st.set_page_config(page_title="Drowsy Driver Detection", layout="wide")
st.title("🚘 Drowsy Driver Detection")
status_box = st.empty()

# Initialize audio alert
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")  # Ensure this file exists

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Function to compute head tilt angle
def get_head_tilt_angle(landmarks, img_w, img_h):
    nose_tip = landmarks[1]
    chin = landmarks[152]

    nose = np.array([nose_tip.x * img_w, nose_tip.y * img_h])
    chin = np.array([chin.x * img_w, chin.y * img_h])

    vector = chin - nose
    vertical = np.array([0, 1])
    angle_rad = np.arccos(np.dot(vector, vertical) / np.linalg.norm(vector))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# OpenCV video stream
cap = cv2.VideoCapture(0)

with st.sidebar:
    st.header("⚙️ Settings")
    drowsy_threshold = st.slider("Tilt Angle Threshold (degrees)", 10, 45, 25)

stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    alert = "Normal"
    alert_color = (0, 255, 0)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            angle = get_head_tilt_angle(landmarks.landmark, img_w, img_h)
            if angle < 90 - drowsy_threshold or angle > 90 + drowsy_threshold:
                alert = "Drowsy! Head Tilt Detected!"
                alert_color = (0, 0, 255)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                pygame.mixer.music.stop()

            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

    cv2.putText(frame, alert, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, 3)
    status_box.markdown(f"### {'🟥' if alert != 'Normal' else '🟩'} {alert}")
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

cap.release()
