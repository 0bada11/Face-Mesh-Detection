import cv2 as cv
import mediapipe as mp
import time

class FaceMesh:
    """
    A class to perform face mesh detection using Mediapipe.

    Attributes:
        static_image_mode (bool): Whether to treat input images as static.
        max_num_faces (int): Maximum number of faces to detect.
        min_detection_confidence (float): Minimum confidence for face detection.
        min_tracking_confidence (float): Minimum confidence for face tracking.
    """

    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initializes the FaceMesh detector with Mediapipe configurations.
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Drawing utilities for rendering landmarks
        self.mpDraw = mp.solutions.drawing_utils  # type: ignore
        self.drawSpaec = self.mpDraw.DrawingSpec(color =(0,255,0),thickness=1, circle_radius=1)

        # Mediapipe FaceMesh model
        self.mpFaceMesh = mp.solutions.face_mesh  # type: ignore
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        # Variables to store results and time
        self.results = None
        self.pTime = 0
        self.cTime = 0

    def findMesh(self, frame, draw=True):
        """
        Detects face landmarks in the given frame and optionally draws them.

        Args:
            frame (np.ndarray): The input video frame (BGR format).
            draw (bool): Whether to draw the landmarks on the frame.

        Returns:
            tuple: (frame with landmarks drawn, list of face landmarks with ID and coordinates)
        """
        # Convert from BGR to RGB as required by Mediapipe
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(frameRGB)

        Mesh = self.results.multi_face_landmarks
        face_sList = []

        if Mesh:
            for facelandmarks in Mesh:
                if draw:
                    # Draw facial tessellation
                    self.mpDraw.draw_landmarks(
                        frame,
                        facelandmarks,
                        # self.mpFaceMesh.FACEMESH_TESSELATION
                        self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpaec,
                        self.drawSpaec
                    )

                # Store all landmarks of this face in a list
                faceList = []
                for id, landmark in enumerate(facelandmarks.landmark):
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    faceList.append([id, x, y])
                face_sList.append(faceList)

        return frame, face_sList

    def showFPS(self, frame):
        """
        Calculates and displays FPS on the current frame.

        Args:
            frame (np.ndarray): The current video frame.

        Returns:
            np.ndarray: Frame with FPS text overlay.
        """
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime) if (self.cTime - self.pTime) > 0 else 0
        self.pTime = self.cTime

        # Draw FPS on frame
        cv.putText(frame, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 255), 2)
        return frame



