import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import LabelEncoder
import gdown
import os
import pickle

file_id = "1E19JGvSKjmcALANyM81cNgbsj2al8MSP"
model_path = "best_model.pkl"

# Check if the model file already exists
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title("Real-Time Behavioral Biometrics Detection")
st.write("Type the following sentence to analyze your typing pattern:")

# Sample text for the user to type
sample_text = "The quick brown fox jumps over the lazy dog."

# Text input area for user typing
user_input = st.text_input("Start typing below:", max_chars=len(sample_text))

# Placeholder to display typing status
status_placeholder = st.empty()

# Lists to capture press and release times
press_times = []
release_times = []

# Function to capture typing dynamics
def capture_typing_dynamics():
    for i, char in enumerate(user_input):
        start_time = time.time()
        while len(user_input) <= i:
            pass
        end_time = time.time()
        press_times.append(start_time)
        release_times.append(end_time)

if st.button("Analyze Typing Pattern"):
    # Capture typing data
    capture_typing_dynamics()

    if len(press_times) > 1:
        # Extract features
        num_keys = len(press_times)
        HD = [release_times[i] - press_times[i] for i in range(num_keys)]
        RPD = [press_times[i + 1] - release_times[i] for i in range(num_keys - 1)]
        PPD = [press_times[i + 1] - press_times[i] for i in range(num_keys - 1)]

        # Create feature vector
        features = np.array([np.mean(HD), np.mean(RPD), np.mean(PPD)]).reshape(1, -1)

        # Make prediction using the pre-trained model
        prediction = model.predict(features)

        # Display the result
        st.write(f"Predicted User: {int(prediction[0])}")
    else:
        st.write("Not enough data captured to analyze typing pattern. Please try again.")
