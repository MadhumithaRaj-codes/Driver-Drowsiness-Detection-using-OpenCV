# Driver Drowsiness Detection using OpenCV

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![IEEE](https://img.shields.io/badge/IEEE-Published-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

A real-time **Driver Drowsiness Detection System** developed using **Python, OpenCV, and Dlib** to improve road safety through intelligent fatigue monitoring and facial landmark analysis.

This project demonstrates practical implementation of **Computer Vision**, **Machine Learning concepts**, and **Advanced Driver Assistance Systems (ADAS)** for real-world safety applications.

---

# Project Highlights

- Developed a real-time fatigue detection system using computer vision techniques
- Implemented **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** algorithms for drowsiness detection
- Integrated webcam-based facial landmark tracking using Dlib
- Built an automated alert mechanism to warn drivers during fatigue events
- Designed for lightweight execution and real-time responsiveness
- Research work published under IEEE ICACCS 2024

---

# System Architecture

```text
Webcam Feed
     ↓
Face Detection
     ↓
Facial Landmark Extraction
     ↓
EAR & MAR Calculation
     ↓
Fatigue Detection Logic
     ↓
Alert Generation
```

---

# Features

## Real-Time Monitoring
Continuously tracks driver facial movements through live webcam input.

## Eye Closure Detection
Uses Eye Aspect Ratio (EAR) calculations to identify prolonged eye closure.

## Yawning Detection
Implements Mouth Aspect Ratio (MAR) analysis to detect yawning patterns associated with fatigue.

## Alert System
Triggers warning alerts when drowsiness thresholds are exceeded.

## Lightweight Implementation
Efficient and suitable for low-resource environments.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Computer vision processing |
| Dlib | Facial landmark detection |
| NumPy | Numerical computations |
| Imutils | Image processing utilities |
| SciPy | Distance calculations |

---

# Project Structure

```bash
Driver-Drowsiness-Detection-using-OpenCV/
│
├── assets/
│   ├── detection.png
│   ├── alert.png
│
├── main.py
├── requirements.txt
├── README.md
├── Driver_Drowsiness_Detection.pdf
└── shape_predictor_68_face_landmarks.dat
```

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/MadhumithaRaj-codes/Driver-Drowsiness-Detection-using-OpenCV.git

cd Driver-Drowsiness-Detection-using-OpenCV
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the Application

```bash
python main.py
```

---

# Detection Workflow

1. Capture live webcam video
2. Detect face using OpenCV
3. Extract facial landmarks using Dlib
4. Compute EAR and MAR values
5. Analyze fatigue indicators
6. Generate alerts if drowsiness is detected

---

# Demo Screenshots

## Real-Time Detection

![Detection](assets/detection.png)

## Drowsiness Alert

![Alert](assets/alert.png)

---

# Applications

- Advanced Driver Assistance Systems (ADAS)
- Automotive Safety Systems
- Smart Transportation
- Driver Monitoring Systems
- Human Behavior Monitoring
- Real-Time Computer Vision Applications

---

# Research Publication

This project is associated with the IEEE ICACCS 2024 conference publication.

📄 Research Paper Included in Repository

---

# Future Enhancements

- Deep learning-based fatigue detection
- Mobile application integration
- Cloud-based monitoring dashboard
- Real-time analytics system
- Night vision support
- AI-powered driver behavior analysis

---

# Why This Project Matters

Driver fatigue contributes to thousands of road accidents every year. This project demonstrates how computer vision and AI-based monitoring systems can improve transportation safety through proactive fatigue detection and alert mechanisms.

---

# Author

## Madhumitha Raj

Master’s Student in Cybersecurity

### Areas of Interest
- Computer Vision
- Artificial Intelligence
- Cybersecurity
- Smart Transportation Systems
- Advanced Driver Assistance Systems (ADAS)

---

# Connect With Me

LinkedIn: YOUR_LINKEDIN_URL

GitHub: https://github.com/MadhumithaRaj-codes

---

# License

This project is intended for academic, educational, and research purposes.
