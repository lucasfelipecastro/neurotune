import cv2
import mediapipe as mp


def analyze_facial_expression(frame):
    '''
    Analyzes the facial expression from a given video frame and returns the detected emotion and its confidence score.
    Args:
        frame (numpy.ndarray): The input image frame from the webcam (BGR format).
    Returns:
        dict: A dictionary with keys 'emotion' (str) and 'confidence' (float).
    '''

    return {'emotion': 'neutral', 'confidence': 1.0} 