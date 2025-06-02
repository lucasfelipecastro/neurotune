import streamlit as st
from app.emotion.nlp.inference import detect_emotion_from_text

st.set_page_config(page_title='Neurotune — Emotion Detection', layout='centered')

st.title('Neurotune — Text Emotion Analyzer')

user_input = st.text_area('Enter your text:', height=150)

if st.button('Analyze Emotion'):
    if user_input.strip() == '':
        st.warning('Please enter some text.')
    else:
        result = detect_emotion_from_text(user_input)
        print('Smile ratio:', result.get('smile_ratio', "N/A"))
        st.success(f"Detected Emotion: **{result['label']}** ({result['score']:.2f})")
