import streamlit as st
import numpy as np
import librosa
import pickle
import os

# Load saved components
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pt.pkl", "rb") as f:
    pt = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("🎵 Real-Time Audio Emotion Recognition")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    feature_vector = np.concatenate([
        mfccs_mean, mfccs_std,
        chroma_mean, chroma_std,
        [zcr_mean, zcr_std],
        [rms_mean, rms_std],
        tonnetz_mean, tonnetz_std,
        contrast_mean, contrast_std
    ])

    return feature_vector

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')  # Allow user to listen to audio

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Extract and preprocess features
        features = extract_features("temp.wav")
        features = features.reshape(1, -1)

        # Correct order: PowerTransform → StandardScale
        features_transformed = pt.transform(features)
        features_scaled = scaler.transform(features_transformed)

        # Predict
        prediction = model.predict(features_scaled)
        predicted_emotion = label_encoder.inverse_transform(prediction)

        st.success(f"🎤 Predicted Emotion: **{predicted_emotion[0].capitalize()}**")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
    finally:
        os.remove("temp.wav")  # Clean up temp file
