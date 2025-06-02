import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Landmark indices for mouth and eyes (MediaPipe Face Mesh)
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_EYE = 33
RIGHT_EYE = 263

# Initialize MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
