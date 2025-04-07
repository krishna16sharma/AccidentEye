from flask import Flask, request, jsonify, render_template
import torch
from torchvision import models, transforms
from werkzeug.utils import secure_filename
from PIL import Image
import io
import os
import cv2
import numpy as np

def detect_accident_in_video(filepath):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    accident_detected = False
    accident_time = None

    consecutive_accident_frames = 0
    threshold = 10  # Number of consecutive accident frames required

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 224x224 for ResNet
        resized = cv2.resize(frame, (224, 224))
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

        if prediction == "Accident":
            consecutive_accident_frames += 1
        else:
            consecutive_accident_frames = 0

        if consecutive_accident_frames >= threshold:
            accident_detected = True
            accident_time = (frame_count - threshold + 1) / fps  # adjust for window offset
            # Save the frame where the accident is detected
            save_path = os.path.join('static', 'accident_frame.jpg')
            cv2.imwrite(save_path, frame)
            break

        frame_count += 1

    cap.release()
    return accident_detected, accident_time if accident_detected else None

# Initialize the Flask application
app = Flask(__name__)

# Load the model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model\\accident_detection_model.pth',map_location=torch.device("cpu")))
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

@app.route('/video')
def video_page():
    return render_template('video.html')  # you'll create this HTML page next

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('static\\uploads', filename)
    file.save(filepath)

    # Process the video and detect accident
    accident_detected, accident_time = detect_accident_in_video(filepath)

    if accident_detected:
        frame_path = '/static/accident_frame.jpg'
        return jsonify({"result": "Accident detected", "time": accident_time, "frame_path": frame_path})
    else:
        return jsonify({"result": "No accident detected"})

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
