import streamlit as st
from keras.models import load_model
from tempfile import NamedTemporaryFile
import os
import logging
import librosa
import numpy as np
from web_scraper import get_random_image_url


class_names=['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 
             'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

def preprocess_audio(file_path):
    audio_file,sr=librosa.load(file_path)
    mfcc_features=librosa.feature.mfcc(y=audio_file,sr=sr, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T,axis=0)
    mfcc_scaled_features=mfcc_scaled_features.reshape(1,-1)
    return mfcc_scaled_features

def predict_audio(preprocessed_data):
    model=load_model('streamlit_model.h5')
    y_predict=model.predict(preprocessed_data)
    class_name=class_names[y_predict.argmax()]
    return class_name


logging.basicConfig(level=logging.INFO)

def audio_classification_page():
    #st.title('Audio Classification App')

    
    #Audio file uploading 
    audio_file = st.file_uploader('Upload an audio file', type=['wav'])
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        with NamedTemporaryFile(delete=False)as tmp_file:
            tmp_filename=tmp_file.name
            tmp_file.write(audio_file.read())
        
         # Log the file path for debugging
        #logging.info("Temporary file path: %s", tmp_filename)

        # Add a button to start the classification
        if st.button('Classify Audio'):
            # Preprocess the audio file
            audio_data = preprocess_audio(tmp_filename)

            # Make predictions using your model
            prediction = predict_audio(audio_data)

            # Display the prediction
            st.header(f"Class audio belongs to is:  {prediction.capitalize()}")

        # Removing the temporary file
        os.remove(tmp_filename)

        