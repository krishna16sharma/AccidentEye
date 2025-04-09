# 🚦 AccidentEye: AI-Powered Accident Detection System

**AccidentEye** is an end-to-end image processing and computer vision system for detecting accidents from images and CCTV video footage. It leverages a pre-trained ResNet50 model for accident classification and includes a web interface built with Flask for easy interaction.

---

## 🎯 Project Highlights

- ✅ Accident detection in **images** using a ResNet50 classifier
- 🎞️ **Video processing** to detect and timestamp accident frames
- ⚙️ Data preprocessing with advanced image augmentation techniques
- 🧠 Powered by **PyTorch**
- 🌐 Interactive **Flask web interface**

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/krishna16sharma/AccidentEye.git
cd AccidentEye
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 💻 Web Interface

- 🌄 **Image Detection**: Upload an image and get an instant prediction.
- 🎥 **Video Detection**: Upload a video. The app returns:
  - Whether an accident occurred
  - Timestamp of the accident (if any)
  - Preview frame where accident occurred

---

## 🧑‍🎓 Developed By

- **[Krishna Sharma S](https://github.com/krishna16sharma)**  
- **[Vaibhav Agrawal](https://github.com/vaibhav-agrawal264)**
- **[Naweed Aquib](https://github.com/aquibn)**
