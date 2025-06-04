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

def analyze_facial_expression(frame):
    '''
    Analyzes the facial expression from a given video frame and returns the detected emotion, its confidence score, and the face bounding box.
    Args:
        frame (numpy.ndarray): The input image frame from the webcam (BGR format).
    Returns:
        dict: A dictionary with keys 'emotion' (str), 'confidence' (float), and 'bbox' (tuple or None).
    '''
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:  # type: ignore
        return {'emotion': 'neutral', 'confidence': 0.0, 'bbox': None}

    face_landmarks = results.multi_face_landmarks[0].landmark  # type: ignore
    h, w, _ = frame.shape

    # Get all landmark points
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    xmin, xmax = int(min(xs)), int(max(xs))
    ymin, ymax = int(min(ys)), int(max(ys))
    bbox = (xmin, ymin, xmax, ymax)

    # Get coordinates for mouth and eyes
    left_mouth = face_landmarks[LEFT_MOUTH]
    right_mouth = face_landmarks[RIGHT_MOUTH]
    left_eye = face_landmarks[LEFT_EYE]
    right_eye = face_landmarks[RIGHT_EYE]

    # Convert normalized coordinates to pixel values
    left_mouth_point = np.array([left_mouth.x * w, left_mouth.y * h])
    right_mouth_point = np.array([right_mouth.x * w, right_mouth.y * h])
    left_eye_point = np.array([left_eye.x * w, left_eye.y * h])
    right_eye_point = np.array([right_eye.x * w, right_eye.y * h])

    # Calculate distances
    mouth_width = np.linalg.norm(right_mouth_point - left_mouth_point)
    eye_distance = np.linalg.norm(right_eye_point - left_eye_point)

    # Smile ratio: higher means more likely to be smiling
    smile_ratio = mouth_width / eye_distance if eye_distance > 0 else 0

    # Simple threshold for smile detection (empirical value)
    if smile_ratio >= 0.51:
        emotion = 'happy'
        confidence = min((smile_ratio - 0.51) * 2, 1.0)
    elif smile_ratio >= 0.20:
        emotion = 'neutral'
        confidence = 1.0 - abs(smile_ratio - 0.35)
    else:
        emotion = 'sad'
        confidence = min((0.20 - smile_ratio) * 5, 1.0)
        
    print(f"Smile ratio: {smile_ratio:.2f}, Confidence: {confidence:.2f}")
    return {'emotion': emotion,  'confidence': float(confidence), 'bbox': bbox}
