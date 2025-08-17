# Face Mesh Detection with Mediapipe & OpenCV

## Overview
This project implements a **Face Mesh Detector** using [Mediapipe FaceMesh](https://developers.google.com/mediapipe/solutions/vision/face_mesh).  
It can detect multiple faces, draw their landmarks (contours or tessellation), and extract the landmark coordinates in real time.

---

## Features
- Real-time face mesh detection with Mediapipe
- Support for multiple faces
- Option to draw landmarks (contours or tessellation)
- Extracts face landmark coordinates (id, x, y)
- FPS display on the video feed

---

## Requirements
Install dependencies:

```bash
pip install opencv-python mediapipe
