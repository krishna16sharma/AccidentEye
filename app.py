from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
import cv2
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model/accident_detection_model.pth',map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define class names
class_names = ["Accident", "Non Accident"]

# Route to render the HTML interface
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the image file
    img = Image.open(file.stream)

    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0)

    # Get the model prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Get the predicted class name
    predicted_class = class_names[predicted.item()]
    
    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
