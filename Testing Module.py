import cv2 as cv
import FaceMesh as fm

"""
Main function to run the webcam and perform real-time face mesh detection.
"""
webcam = cv.VideoCapture(0)  # Start video capture from webcam
detector = fm.FaceMesh()        # Create FaceMesh detector object

while True:
    # Read a frame and flip it horizontally (mirror view)
    isTrue, frame = webcam.read()
    frame = cv.flip(frame, 1)
    # Detect face mesh and optionally draw
    frame, faceList = detector.findMesh(frame)

    # Print coordinates of face mesh points if any faces are detected
    if len(faceList) != 0:
        print(faceList)

    # Show FPS on frame
    frame = detector.showFPS(frame)

    # Display the result
    cv.imshow("Face Detection", frame)

    # Exit if 'Esc' is pressed
    if cv.waitKey(1) & 0xFF == 27:
        break

# Release resources
webcam.release()
cv.destroyAllWindows()


