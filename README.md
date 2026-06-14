EMOTION DETECTION: Real-Time Facial Emotion Recognition 


![Python](https://img.shields.io/badge/Python-3.10.11-blue.svg) 

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg) 

![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red.svg) 

![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-green.svg) 

![License](https://img.shields.io/badge/License-MIT-lightgrey.svg) 



Project Overview 
EMOTION DETECTION is a lightweight, edge-deployable deep learning system engineered 
to classify seven universal human emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, 
Surprise) from facial images in real-time.  

Built as a Master of Computer Applications (MCA) thesis project, this system leverages a 
customized MobileNetV2 architecture utilizing a multi-phase transfer learning strategy. It is 
specifically optimized to run locally on consumer-grade hardware (NVIDIA RTX 3060) 
without relying on high-latency cloud APIs, achieving sub-100ms inference speeds via 
native OpenCV hardware hooks and a custom Streamlit UI. 


Core Features 

Real-Time Edge Inference: Bypasses browser-based latency by utilizing 
`cv2.VideoCapture(0)` for zero-latency optical tracking. 

Lightweight Architecture: Utilizes MobileNetV2 with depthwise separable convolutions to 
drastically reduce parameter count and VRAM footprint.
 
Dual-Tab UI/UX: Features a custom "Cyberpunk" aesthetic Streamlit dashboard with 
dedicated modules for live webcam scanning and static image diagnostics. 

Advanced Regularization: Trained on the unconstrained FER2013-4X dataset using real
time spatial data augmentation and heavy dropout layers to prevent overfitting.

Peak Performance: Achieves 63.34% validation accuracy, approaching the accepted 
human-level baseline for the highly ambiguous FER2013 benchmark. 


Technology Stack & Environment 

To ensure strict scientific reproducibility and hardware stability (specifically native 
Windows CUDA/cuDNN support without WSL2), this project is locked to the following 
environment: 

Language: Python 3.10.11 

Deep Learning: TensorFlow 2.10.0 / Keras 

Computer Vision: OpenCV 4.13.0 (Haar Cascades) 

Frontend UI: Streamlit 1.22.0 

Data Science: NumPy 1.24.3, Pandas, Matplotlib, Seaborn 

System Architecture & Directory Structure 

EMOTION_DETECTION_PROJECT/ 
│ 
├── dataset/ 
│   ├── train/                              # Contains training images across 7 classes               
│   └── test/                               # Contains isolated validation images 
│ 
├── models/              
│   ├── Emotionfinalmodel.h5                # Phase 3 Fine-Tuned Weights 
│   ├── Emotionlabels.json                  # Categorical Mapping 
│   └── haarcascade_frontalface_default.xml 
│ 
├── src/ 
│   ├── eda_analysis.ipynb                   # Exploratory Data Analysis & Visualizations 
│   └── model_training.py                    # Data generators, MobileNetV2 architecture, callbacks 
│ 
├── App.py                                     # Main Streamlit Execution Script 
├── requirements.txt                           # Environment dependencies 
└── README.md                                  # Project Documentation 
             
Installation & Setup 


1. Clone the repository and navigate to the project root: 

Bash
 
git clone [https://github.com/Vishubhaumick/Emotion
Detection.git](https://github.com/Vishubhaumick/Emotion-Detection.git) 

cd Emotion-Detection 

2. Initialize and activate a strict virtual environment: 

Bash 

python -m venv venv 

On Windows: 

venv\Scripts\activate 

On macOS/Linux: 

source venv/bin/activate 

3. Install the locked dependencies to prevent version conflicts: 

Bash 

pip install -r requirements.txt 

4. Verify GPU Allocation (Optional but recommended): Ensure your local CUDA 11.2 and 
cuDNN 8.1.0 paths are configured correctly for TensorFlow 2.10.0 to recognize your GPU. 

Execution 

To launch the EMOTION DETECTION dashboard, run the master execution script: 

Bash 

streamlit run App.py 

Note: Ensure your physical webcam is not currently locked by another application (e.g., 
Zoom, Teams) before initializing the Real-Time Sensor tab. 
Evaluation & Metrics 

The multi-phase fine-tuning strategy yielded the following progression on the 
mathematically isolated validation vault: 

• Phase 1 (Frozen Base): ~47.99% Accuracy 

• Phase 2 (Mid-Level Unfreeze): ~59.11% Accuracy 

• Phase 3 (Deep Fine-Tuning): 63.34% Peak Accuracy (Epoch 23) 

Comprehensive diagnostic charts, including Confusion Matrices and Epoch progression 
graphs, are available in the project report documentation.