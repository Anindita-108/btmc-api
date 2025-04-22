from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('btmc.h5')  # Load model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input_data']  # Get data from frontend
    prediction = model.predict(np.array(data).reshape(1, -1))  # Reshape if needed
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render needs this