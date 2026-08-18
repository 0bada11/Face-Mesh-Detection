import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils # type: ignore
mpFaceMesh = mp.solutions.face_mesh # type: ignore
faceMesh = mpFaceMesh.FaceMesh()
# 
drawSpaec = mpDraw.DrawingSpec(thickness=1 , circle_radius =1)

webcam = cv.VideoCapture(0)

while True:
    # Read a frame from webcam and flip horizontally
    isTrue, frame = webcam.read()
    frame = cv.flip(frame, 1)
    
    # convert from BGR to RGB
    frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)

    # store the results in one variable:
    Mesh = results.multi_face_landmarks

    # display the landmarks
    if Mesh:
        for facelandmarks in Mesh:
            mpDraw.draw_landmarks(frame,facelandmarks,mpFaceMesh.FACEMESH_TESSELATION,drawSpaec,drawSpaec) # tye also mpFaceMesh.FACEMESH_CONTOURS
            for id, landmark in enumerate(facelandmarks.landmark):
                # print(landmark)
                h, w, c = frame.shape
                x,y = int(landmark.x*w) , int(landmark.y*h)
                print(id,x,y)


    # Show the video feed
    cv.imshow("Face Detection", frame)

    # Break loop when 'Esc' key is pressed
    if cv.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv.destroyAllWindows()