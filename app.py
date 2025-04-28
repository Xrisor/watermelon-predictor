from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
from PIL import Image
import cv2
import rembg

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Helper function: Preprocess image (resize, remove background, extract features)
def preprocess_image(image_path):
    # Open image
    img = Image.open(image_path).convert("RGB")
    
    # Remove background
    output = rembg.remove(img)
    
    # Resize to 100x100
    img_resized = output.resize((100, 100))
    
    # Convert to array
    img_array = np.array(img_resized)
    
    # Extract Color Feature (average RGB)
    R_mean = np.mean(img_array[:, :, 0])
    G_mean = np.mean(img_array[:, :, 1])
    B_mean = np.mean(img_array[:, :, 2])

    # Convert to grayscale for shape analysis
    img_gray = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            roundness = 0
        else:
            roundness = 4 * np.pi * (area / (perimeter ** 2))
    else:
        roundness = 0

    features = np.array([[R_mean, G_mean, B_mean, roundness]])
    return features

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        features = preprocess_image(filepath)
        prediction = model.predict(features)[0]

        label = 'Sweet 🍉' if prediction == 1 else 'Unsweet 🍉'

        os.remove(filepath)  # Clean up uploaded file after prediction

        return render_template('index.html', prediction=label)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
