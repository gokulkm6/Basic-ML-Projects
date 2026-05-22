
# 🤘 Hand Gesture Recognition using DenseNet201 (Transfer Learning)

This repository contains an advanced image classification model designed to recognize **hand gestures** using transfer learning with the **DenseNet201** architecture. The model is trained to classify 10 different gesture categories using a custom image dataset and extensive data augmentation techniques.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Classes Supported](#classes-supported)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Training Details](#training-details)
- [Performance](#performance)
- [Usage Guide](#usage-guide)
- [Model Deployment](#model-deployment)
- [Future Work](#future-work)
- [License](#license)

---

## 🎯 Project Overview

Hand gestures are a powerful form of non-verbal communication. This project builds a robust gesture classification model using TensorFlow and transfer learning. The model learns to classify static hand gesture images from a labeled dataset into one of ten predefined categories.

---

## 📁 Dataset Structure

Your dataset should follow this folder structure:

```
images/
├── call_me/
├── fingers_crossed/
├── okay/
├── paper/
├── peace/
├── rock/
├── rock_on/
├── scissor/
├── thumbs/
└── up/
```

- Each subfolder must contain image files of that specific gesture.
- Images are resized to **60x60 RGB** format before training.

---

## ✌️ Classes Supported

The model supports recognition of the following 10 gestures:

```
['call_me', 'fingers_crossed', 'okay', 'paper', 'peace',
 'rock', 'rock_on', 'scissor', 'thumbs', 'up']
```

---

## 🏗️ Model Architecture

- **Base Model**: DenseNet201 (without top layers, pretrained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Dense(128, ReLU)
  - Dense(10, Softmax)

```python
pretrained_model = DenseNet201(include_top=False, weights='imagenet', pooling='avg')
x = Dense(128, activation='relu')(pretrained_model.output)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=pretrained_model.input, outputs=output)
```

---

## 🛠️ Technologies Used

| Component       | Version         |
|----------------|------------------|
| Language        | Python 3.9+      |
| Libraries       | TensorFlow, Keras, NumPy, Pandas, Matplotlib |
| Data Augmentation | ImageDataGenerator |
| Evaluation      | Classification Report, Accuracy, Log Loss |

---

## 📊 Training Details

- **Epochs**: 20  
- **Batch Size**: 32  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Augmentation**: Rotation, Flip, Zoom, Shear, Shift  
- **Validation Split**: from test set

### Training Progress (Sample)

| Epoch | Accuracy | Val Accuracy | Val Loss |
|-------|----------|--------------|----------|
| 1     | 38.94%   | 26.75%       | 11.76    |
| 10    | 92.80%   | 86.25%       | 0.96     |
| 20    | 95.58%   | 97.37%       | 0.07     |

---

## 🧪 Performance

- **Validation Accuracy (Final)**: ~97.37%
- **Prediction Sample**:  
  Input: `rock_on/1317.jpg`  
  Output: `rock_on`

---

## 🚀 Usage Guide

### 📦 Setup

```bash
pip install tensorflow pandas numpy matplotlib
```

### 🏁 Train the Model

Ensure the folder `images/` is placed correctly and run the script or notebook to start training.

### 🔍 Predict Custom Image

```python
image = load_img('path/to/image.jpg', target_size=(60,60))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
predicted_label = np.argmax(prediction)
```

---

## 💾 Model Deployment

You can save and reload the model:

```python
model.save('handgest.hdf5')
model = keras.models.load_model('handgest.hdf5')
```

Ideal for embedding in a web/mobile app using TensorFlow Lite or TF.js for real-time gesture recognition.

---

## 🔮 Future Work

- Integrate with live webcam feed for real-time recognition
- Add more gesture classes and fine-tune for different skin tones/lighting
- Convert to TensorFlow Lite for mobile deployment
- Build a Flask-based API for serving predictions

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> Developed with 💻 by [@gokulkm6](https://github.com/gokulkm6)
