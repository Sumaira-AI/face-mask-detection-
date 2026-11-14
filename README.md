
# Face Mask Detection & Image Processing System

An Interactive Computer Vision and Deep Learning Application using Python, OpenCV, YOLO, and Streamlit.

## 1. Project Overview

The AI-Powered Face Mask Detection System is an interactive computer vision application that combines traditional image processing techniques with deep learning-based object detection.

The application allows users to upload an image, apply multiple image preprocessing and enhancement techniques, compare the original and processed images, and select either image for further analysis.

After image selection, the system performs image analysis, generates histogram visualizations, and applies a trained YOLO model to detect face masks.

The main purpose of this project is to explore how different computer vision techniques and image preprocessing operations can be integrated with an AI-based detection model in a single interactive application.

## 2. Problem Statement

Manual identification and monitoring of face mask usage can be time-consuming, especially when dealing with multiple images.

Image quality, brightness, contrast, noise, and other visual conditions can also affect how images are interpreted by computer vision systems.

This project provides an interactive platform where users can experiment with image preprocessing techniques, analyze image properties, and perform automated face mask detection using a trained YOLO model.

## 3. Project Objectives

- Implement multiple computer vision image processing techniques.
- Improve and analyze image quality through preprocessing.
- Compare original and processed images.
- Allow users to select their preferred image for detection.
- Generate histogram and pixel intensity visualizations.
- Implement YOLO-based face mask detection.
- Combine image processing and deep learning in an interactive web application.

## 4. Computer Vision Techniques Implemented

The project incorporates multiple image processing and enhancement techniques.

### A. Image Transformation and Resizing

- Image Resizing
- Image Zooming

### B. Image Smoothing and Noise Reduction

- Gaussian Blur
- Average Blur
- Median Blur
- Bilateral Filtering

### C. Image Enhancement

- Image Sharpening
- Brightness Adjustment
- Contrast Adjustment
- Histogram Equalization

### D. Edge Detection

- Sobel Edge Detection
- Canny Edge Detection

These techniques allow users to explore different image processing operations and observe their effects on the input image.

## 5. Interactive Image Comparison

One of the main features of this application is its two-image comparison interface.

The application displays:

**1. Original Image:** The image uploaded by the user without preprocessing.

**2. Processed Image:** The image after applying selected computer vision preprocessing techniques.

Users can visually compare both images and select either the original or processed version for further analysis and detection.

## 6. Image Analysis and Visualization

After selecting an image, the application provides image analysis and visualization features, including:

- Histogram Visualization
- Pixel Intensity Analysis
- Image Processing Results
- Edge Detection Visualization

These visualizations help users understand changes in image characteristics after applying different preprocessing techniques.

## 7. YOLO-Based Face Mask Detection

The application integrates a trained YOLO deep learning model for automated face mask detection.

The selected image is passed to the trained model, which performs object detection and displays the resulting predictions.

### Detection Features

- AI-based object detection
- Bounding box visualization
- Detection confidence scores
- Detection result display

The trained model is stored in:

`model/best.pt`

## 8. Application Workflow

1. Upload an input image.
2. Display the original image.
3. Apply computer vision preprocessing techniques.
4. Display the processed image alongside the original.
5. Compare both images.
6. Select either the original or processed image.
7. Perform image analysis and generate histograms.
8. Apply the trained YOLO model.
9. Display face mask detection results.

## 9. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and computer vision |
| YOLO | Deep learning object detection |
| Streamlit | Interactive web application |
| NumPy | Numerical and image array operations |
| Matplotlib | Data visualization and histograms |
| PyTorch | Deep learning framework |

## 10. Project Structure

```text
facemask_app/
│
├── model/
│   └── best.pt
│
├── app.py
├── detector.py
├── processing.py
├── requirements.txt
└── README.md
```

## 11. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sumaira-AI/face-mask-detection-.git
```

### Step 2: Navigate to the Project Directory

```bash
cd face-mask-detection-/facemask_app
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser through the local Streamlit server.

## 12. Project Significance

This project demonstrates the integration of traditional computer vision techniques with modern deep learning approaches.

It provides practical experience in:

- Image preprocessing and enhancement
- Noise reduction and filtering
- Edge detection
- Image histogram analysis
- Deep learning model integration
- Object detection
- Interactive AI application development

The image comparison feature also provides an opportunity to study how different preprocessing operations influence the visual input supplied to an object detection model.

## 13. Future Improvements

- Live webcam-based face mask detection.
- Video-based face mask monitoring.
- Performance comparison between original and preprocessed images.
- Additional image enhancement techniques.
- Improved detection performance and model evaluation.

## 14. Developer

**Sumaira Shabir**

GitHub: [Sumaira-AI](https://github.com/Sumaira-AI)

---

**Project Category:** Computer Vision | Deep Learning | AI-Based Object Detection

**Project Type:** Interactive Web Application
