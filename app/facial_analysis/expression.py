import cv2
import numpy as np
import mediapipe.python.solutions.face_mesh as mp_face_mesh

# Landmark indices for mouth and eyes (MediaPipe Face Mesh)
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_EYE = 33
RIGHT_EYE = 263
