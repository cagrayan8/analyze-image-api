from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import urllib.parse 
import json
import base64
import traceback
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
import firebase_admin
from firebase_admin import credentials, storage as firebase_storage

app = Flask(__name__)
print("✅ MODE: Using keras built-in weights, skipping .h5 download.")

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Modeli yükle
model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    alpha=1.0,
    input_shape=(160, 160, 3)
)

def predict_features(image_array):
    return model(image_array, training=False)



def extract_features(image_url):
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = image.resize((160, 160))
        image_array = np.expand_dims(np.array(image), axis=0)
        image_array = preprocess_input(image_array)
        features = predict_features(image_array)
        return features.numpy(), None
    except Exception as e:
        return None, str(e)

def extract_features_from_blob(blob):
    try:
        image_bytes = blob.download_as_bytes()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = image.resize((160, 160))
        image_array = np.expand_dims(np.array(image), axis=0)
        image_array = preprocess_input(image_array)
        features = predict_features(image_array)
        return features.numpy(), None
    except Exception as e:
        return None, str(e)

@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "API is running",
        "model_loaded": model is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    image_url_1 = data.get('image_url_1')
    image_url_2 = data.get('image_url_2')

    if not image_url_1 or not image_url_2:
        return jsonify({'error': 'Both image_url_1 and image_url_2 are required'}), 400

    try:
        features1, err1 = extract_features(image_url_1)
        features2, err2 = extract_features(image_url_2)

        if err1:
            return jsonify({'error': f'Image 1 processing failed: {err1}'}), 500
        if err2:
            return jsonify({'error': f'Image 2 processing failed: {err2}'}), 500

        similarity = cosine_similarity(features1, features2)[0][0]
        similarity = float(np.clip(similarity, 0.0, 1.0))
        similarity_percent = round(similarity * 100, 2)

        return jsonify({
            'similarity': similarity_percent,
            'status': 'success'
        })
    except Exception as e:
        print("❌ EXCEPTION CAUGHT IN /analyze")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Firebase Admin başlat
if not firebase_admin._apps:
    cred_json = base64.b64decode(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']).decode('utf-8')
    cred = credentials.Certificate(json.loads(cred_json))
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'myfamilyapp-9a733.firebasestorage.app'
    })

@app.route('/analyze_family', methods=['POST'])
def analyze_family():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    image_url = data.get('image_url')
    family_id = data.get('family_id')

    if not image_url or not family_id:
        return jsonify({'error': 'image_url and family_id required'}), 400

    try:
        # Yüklenen görselin özelliklerini çıkar
        uploaded_features, err = extract_features(image_url)
        if err or uploaded_features is None:
            return jsonify({'error': f'Uploaded image failed: {err}'}), 500

        bucket = firebase_storage.bucket()
        prefix = f"assignments_images/{family_id}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Hiç görsel yoksa, ilk yükleme
        if len(blobs) == 0:
            return jsonify({
                'max_similarity': 0.0,
                'status': 'first_upload'
            })

        max_similarity = 0.0
        # URL'den blob adını çıkar
        parsed_url = urllib.parse.urlparse(image_url)
        # Firebase Storage URL formatını ayrıştır
        path_parts = parsed_url.path.split('/o/')
        if len(path_parts) < 2:
            return jsonify({'error': 'Invalid image URL format'}), 400
            
        encoded_blob_name = path_parts[1].split('?')[0]
        uploaded_blob_name = urllib.parse.unquote(encoded_blob_name)
        
        for blob in blobs:
            # URL'den çıkarılan blob adı ile karşılaştır
            if blob.name == uploaded_blob_name:
                continue
                
            print(f"🔍 Comparing uploaded image with: {blob.public_url}")

            compare_features, err2 = extract_features_from_blob(blob)
            if err2 or compare_features is None:
                continue

            similarity = cosine_similarity(uploaded_features, compare_features)[0][0]
            similarity = float(np.clip(similarity, 0.0, 1.0))
            max_similarity = max(max_similarity, similarity)

        return jsonify({
            'max_similarity': round(max_similarity * 100, 2),
            'status': 'success'
        })

    except Exception as e:
        print("❌ EXCEPTION CAUGHT IN /analyze_family")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500