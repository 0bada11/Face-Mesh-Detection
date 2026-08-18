# Face Mesh Detection

> Real-time 468-point facial landmark detection using MediaPipe Face Mesh and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00A67E?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project detects faces in a video stream and extracts 468 facial landmarks from each
one, drawing the facial contours over the live feed. The landmark coordinates are returned
as ordinary pixel values, so they can feed straight into downstream work: expression
analysis, face filters, blink detection, head-pose estimation, or AR overlays.

The detector is packaged as a reusable `FaceMesh` class rather than a single script, so it
can be imported into other projects.

## Features

- 468 facial landmarks per face, tracked in real time
- Multi-face support (defaults to 2, configurable)
- Contour rendering with a configurable `DrawingSpec`
- Landmark coordinates returned as `[id, x, y]` pixel lists, one list per face
- Live FPS counter
- Adjustable detection and tracking confidence thresholds

## Requirements

- Python 3.8 or newer
- A webcam, or a video file to process

## Installation

```bash
git clone https://github.com/0bada11/face-mesh-detection.git
cd face-mesh-detection
pip install -r requirements.txt
```

## Usage

Run the demo:

```bash
cd src
python demo.py
```

Or import the detector into your own code:

```python
import cv2 as cv
import face_mesh as fm

detector = fm.FaceMesh(max_num_faces=2)
cap = cv.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame, faces = detector.findMesh(frame, draw=True)

    if faces:
        # faces[0] is a list of [id, x, y] for the first detected face
        print(len(faces[0]), "landmarks")

    frame = detector.showFPS(frame)
    cv.imshow("Face Mesh", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
```

Press `Esc` to quit.

### API

| Member | Description |
| --- | --- |
| `FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)` | Construct the detector |
| `findMesh(frame, draw=True)` | Returns `(frame, faces)` where `faces` is a list of per-face `[id, x, y]` landmark lists |
| `showFPS(frame)` | Draws the current frame rate onto the frame and returns it |

## How It Works

MediaPipe Face Mesh runs a two-stage pipeline. A lightweight detector first locates faces in
the frame, then a regression model predicts 468 3D landmark positions across the face
surface. Because the second stage tracks between frames rather than re-detecting every time,
the pipeline stays fast enough for live video on CPU.

The landmarks MediaPipe returns are normalised to the `0.0–1.0` range. This project converts
them to pixel coordinates by multiplying against the frame width and height, which is the
form most downstream OpenCV work expects.

By default the drawing step uses `FACEMESH_CONTOURS`, which outlines the eyes, brows, lips,
and face oval. Switching that to `FACEMESH_TESSELATION` in `findMesh` renders the full
triangulated mesh instead. Either way, all 468 landmarks are returned in the coordinate list.

## Project Structure

```
face-mesh-detection/
├── src/
│   ├── face_mesh.py          # Reusable FaceMesh detector class
│   ├── face_mesh_basics.py   # Minimal implementation, no abstraction
│   └── demo.py               # Example usage of the detector class
├── requirements.txt
└── LICENSE
```

`face_mesh_basics.py` is the unrefactored version, kept because it shows the raw MediaPipe
calls without the wrapper class in the way.

## Related Projects

See [Computer Vision Fundamentals](https://github.com/0bada11/computer-vision-fundamentals)
for the hand tracking, pose estimation, and face detection modules built the same way.

## License

Released under the [MIT License](LICENSE).
