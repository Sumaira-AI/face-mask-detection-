# 😷 Face Mask Detection System

## 📌 Project Overview
The **Face Mask Detection System** is a computer vision application that uses a YOLO deep learning model to detect face masks in images.

The system is designed to identify face mask usage through an AI-powered detection approach and provide a simple interface for interacting with the model.

## 🎯 Problem Statement
Wearing face masks can help reduce the spread of respiratory infections. However, manually monitoring mask usage in public places is time-consuming.

This project explores how computer vision and deep learning can automate face mask detection and support faster visual monitoring.

## 🚀 Technologies Used
- **Python** – Core programming language
- **YOLO** – Object detection model
- **OpenCV** – Image processing
- **Streamlit** – Web application interface
- **PyTorch** – Deep learning framework

## ✨ Key Components
- AI-based face mask detection
- Trained YOLO model integration
- Image processing using OpenCV
- Interactive web interface using Streamlit

## 📂 Project Structure

```text
facemask_app/
│
├── model/
│   └── best.pt
│
├── app.py
├── detector.py
├── processing.py
├── README.md
└── requirements.txt
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sumaira-AI/face-mask-detection-.git
```

### 2. Navigate to the Project Folder

```bash
cd face-mask-detection-/facemask_app
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

## 🤖 Trained Model
The project uses a trained YOLO model stored in:

```text
model/best.pt
```

Make sure the trained model file is available before running the application.

## 🎓 Project Purpose
This project demonstrates the practical application of artificial intelligence, deep learning, and computer vision in automated object detection.

---

**Developed by:** Sumaira Shabir

**GitHub:** [Sumaira-AI](https://github.com/Sumaira-AI)

