import streamlit as st
import numpy as np
import cv2
from app.facial_analysis.expression import analyze_facial_expression

st.set_page_config(page_title='Neurotune — Webcam Emotion Detection', layout='centered')
st.title('🎧 Neurotune — Webcam Emotion Analyzer')

st.write('Capture an image from your webcam to detect your facial emotion.')

img_file_buffer = st.camera_input('Take a picture')

if img_file_buffer is not None:
    # Convert the uploaded image to a numpy array (BGR format for OpenCV)
    file_bytes = np.asarray(bytearray(img_file_buffer.getvalue()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Analyze facial expression
    result = analyze_facial_expression(frame)
    st.success(f"Detected Emotion: **{result['emotion']}** (Confidence: {result['confidence']:.2f})") 