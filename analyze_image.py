from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os  # Eksik olan os modülü eklendi

app = Flask(__name__)

# GPU bellek optimizasyonu
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Model tanımı
model = MobileNetV2(
    weights=None,
    include_top=False,
    pooling='avg',
    alpha=0.35,
    input_shape=(96, 96, 3)
)

# Ağırlıkları yükle
try:
    model.load_weights("models/mobilenet.h5")
    print("✅ Model weights loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    # Hata durumunda uygulamayı durdur
    raise SystemExit(1)

# Tahmin fonksiyonu
@tf.function
def predict_features(image_array):
    return model(image_array, training=False)

# Özellik çıkarımı
def extract_features(image_url):
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = image.resize((96, 96))
        
        image_array = np.expand_dims(np.array(image), axis=0)
        image_array = preprocess_input(image_array)
        
        features = predict_features(image_array)
        return features.numpy(), None
    except Exception as e:
        return None, str(e)

# Health check endpoint
@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "API is running",
        "model_loaded": model is not None
    })

# API endpoint
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

        similarity = float(cosine_similarity(features1, features2)[0][0]) * 100
        return jsonify({
            'similarity': round(similarity, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Ana uygulama başlatma
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)