from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model = MobileNetV2(
    weights=None,
    include_top=False,
    pooling='avg',
    alpha=0.35,
    input_shape=(96, 96, 3)
)
model.load_weights("models/mobilenet.h5")


@tf.function(experimental_relax_shapes=True)
def predict_features(image_array):
    return model(image_array, training=False)

def extract_features(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = image.resize((96, 96))
        image_array = np.expand_dims(np.array(image), axis=0)
        image_array = preprocess_input(image_array)
        features = predict_features(image_array)
        return features.numpy(), None
    except Exception as e:
        return None, str(e)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_url_1 = data.get('image_url_1')
    image_url_2 = data.get('image_url_2')

    if not image_url_1 or not image_url_2:
        return jsonify({'error': 'Both image_url_1 and image_url_2 are required'}), 400

    features1, err1 = extract_features(image_url_1)
    features2, err2 = extract_features(image_url_2)

    if err1 or err2:
        return jsonify({'error': err1 or err2}), 500

    similarity = cosine_similarity(features1, features2)[0][0] * 100
    return jsonify({'similarity': round(similarity, 2)})

@app.before_first_request
def log_memory_usage():
    mem = psutil.Process().memory_info().rss / 1024 ** 2
    print(f"Current memory usage: {mem:.2f} MB")

# Local test için (Railway değilken çalışır)
if __name__ == '__main__':
    app.run(debug=True)
