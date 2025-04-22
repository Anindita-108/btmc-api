from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('btmc.h5')  # Load your model

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if image was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # 2. Process image
    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((256, 256))  # Adjust size to match your model's input
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    # 3. Predict and format results
    prediction = model.predict(img_array)
    class_names = ["Pituitary", "Glioma", "Meningioma", "Non-Tumour"]  # MUST match model's output order
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
        "tumor_type": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "Pituitary": float(prediction[0][0]),
            "Glioma": float(prediction[0][1]),
            "Meningioma": float(prediction[0][2]),
            "Non-Tumour": float(prediction[0][3])
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render-compatible settings
