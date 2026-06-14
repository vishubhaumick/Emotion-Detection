import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Emotion Detection",
    layout="wide"
)

@st.cache_resource
def load_emotion_model():
    model = load_model(r'E:\Emotion Detection\emotion_final_model.h5')
    with open(r'E:\Emotion Detection\emotion_labels.json', 'r') as f:
        labels = json.load(f)
    return model, labels

model, emotion_labels = load_emotion_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_face(face_img):
    face_resized    = cv2.resize(face_img, (96, 96))
    face_rgb        = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    face_input      = np.expand_dims(face_normalized, axis=0)
    return face_input

def predict_emotion(face_input):
    prediction  = model.predict(face_input, verbose=0)[0]
    emotion_idx = np.argmax(prediction)
    emotion     = emotion_labels[emotion_idx]
    confidence  = prediction[emotion_idx] * 100
    return emotion, confidence, prediction

def draw_results(image, faces, emotions):
    result_img = image.copy()
    for (x, y, w, h), (emotion, confidence, prediction) in zip(faces, emotions):
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(
            result_img, label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )
    return result_img

# --- UI LAYOUT ---
st.title("Emotion Detection from Face")

st.sidebar.title("About")
st.sidebar.write("Model    : MobileNetV2")
st.sidebar.write("Dataset  : FER2013 4X")
st.sidebar.write("Accuracy : 63.34%")
st.sidebar.write("Emotions : 7 classes")
st.sidebar.write("")
st.sidebar.write("Emotions detected:")
for label in emotion_labels:
    st.sidebar.write(f"  - {label.capitalize()}")

# Create the tabs
tab1, tab2 = st.tabs(["🔴 Live Webcam Feed", "📁 Upload Image"])

# --- TAB 1: LIVE WEBCAM (NATIVE OPENCV) ---
with tab1:
    st.subheader("Real-Time Detection")
    st.write("Check the box below to start your webcam. Uncheck to stop.")
    
    run_webcam = st.checkbox("Start Live Webcam")
    FRAME_WINDOW = st.image([])
    
    if run_webcam:
        # 0 is usually the default built-in webcam
        cap = cv2.VideoCapture(0)
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam. Please check your permissions.")
                break
            
            # Convert BGR (OpenCV) to GRAY for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process faces
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                
                if face_crop.size == 0:
                    continue
                    
                face_input = preprocess_face(face_crop)
                emotion, confidence, _ = predict_emotion(face_input)
                
                # Draw boxes and text on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion} ({confidence:.1f}%)"
                cv2.putText(
                    frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )
            
            # Convert final frame from BGR to RGB for Streamlit rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
        cap.release()

# --- TAB 2: STATIC IMAGE UPLOAD ---
with tab2:
    st.subheader("Static Image Analysis")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            st.warning("No face detected! Please upload a clear front facing image.")
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

        else:
            all_emotions = []
            for (x, y, w, h) in faces:
                face_crop  = image[y:y+h, x:x+w]
                face_input = preprocess_face(face_crop)
                emotion, confidence, prediction = predict_emotion(face_input)
                all_emotions.append((emotion, confidence, prediction))

            result_image = draw_results(image_rgb, faces, all_emotions)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Result")
                st.image(result_image, use_column_width=True)

            with col2:
                st.subheader("Emotion Analysis")
                for i, (emotion, confidence, prediction) in enumerate(all_emotions):
                    st.write(f"Face {i+1}:")
                    st.write(f"  Emotion    : {emotion.upper()}")
                    st.write(f"  Confidence : {confidence:.2f}%")
                    st.write("")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors  = [
                        'red' if j == np.argmax(prediction) else 'steelblue'
                        for j in range(len(emotion_labels))
                    ]
                    ax.barh(emotion_labels, prediction * 100, color=colors)
                    ax.set_xlabel('Confidence %')
                    ax.set_title(f'Face {i+1} Emotion Probabilities')
                    ax.set_xlim(0, 100)
                    st.pyplot(fig)
                    plt.close()