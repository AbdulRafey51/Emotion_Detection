from flask import Flask, request, jsonify
import numpy as np
import librosa
import pickle
import os
from werkzeug.utils import secure_filename
from collections import Counter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load saved model and preprocessing tools
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pt.pkl", "rb") as f:
    pt = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Helper function: chunk audio into 3-second overlapping segments
def chunk_audio(y, sr, chunk_duration=3, overlap=1):
    chunk_samples = int(chunk_duration * sr)
    step = int((chunk_duration - overlap) * sr)
    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, step):
        chunks.append(y[start:start + chunk_samples])
    return chunks

# Feature extraction function
def extract_features_from_chunk(y, sr):
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

# API endpoint
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part. Please upload a .wav file as 'audio'."}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            y, sr = librosa.load(file_path)

            chunks = chunk_audio(y, sr, chunk_duration=3, overlap=1)
            emotions = []

            for chunk in chunks:
                features = extract_features_from_chunk(chunk, sr).reshape(1, -1)
                features_transformed = pt.transform(features)
                features_scaled = scaler.transform(features_transformed)
                prediction = model.predict(features_scaled)
                emotion = label_encoder.inverse_transform(prediction)[0].capitalize()
                emotions.append(emotion)

            # Return majority emotion or all chunk predictions
            most_common = Counter(emotions).most_common(1)[0][0]

            return jsonify({
                "emotion": most_common,
                "all_chunk_emotions": emotions
            }), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.remove(file_path)

    else:
        return jsonify({"error": "Invalid file type. Only .wav supported."}), 400

if __name__ == '__main__':
    app.run(debug=True)
