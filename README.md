# 🌿 PlantCare AI

AI-powered plant disease detection system built using **PyTorch, Transfer Learning, EfficientNet-B0, Optuna and Streamlit**. PlantCare AI can identify plant diseases from leaf images across **38 different classes** and provides the predicted disease, confidence score and Top-3 predictions through a clean web interface.

## 🚀 Features

- 🌱 Plant disease classification
- 🧠 EfficientNet-B0 Transfer Learning
- 🎯 38 disease and healthy classes
- ⚙️ Optuna hyperparameter optimization
- 🛑 Early stopping
- 📊 Accuracy, Precision, Recall and F1 Score
- 🔥 Confusion Matrix
- 🏆 Top-3 predictions
- 📈 Confidence scores
- 🖥️ Streamlit web application
- ⚡ GPU-supported inference

## 🎯 Problem Statement

Plant diseases can significantly affect crop production and quality. Identifying diseases manually can require expert knowledge and time. This project aims to automate plant disease identification using deep learning and computer vision by analyzing plant leaf images and predicting the corresponding disease or healthy condition.

## 🧠 Model

The final model uses **EfficientNet-B0 pretrained on ImageNet** with Transfer Learning. The pretrained feature extraction layers are used to learn meaningful visual patterns from plant leaf images, while the final classifier is customized for the 38 classes in the dataset.

### Model Pipeline

Plant Leaf Image → Image Preprocessing → EfficientNet-B0 → Feature Extraction → Classifier → 38 Classes → Prediction + Confidence

## 🔬 Development Pipeline

The project was developed through multiple stages including a custom CNN baseline, data augmentation, transfer learning with EfficientNet-B0, early stopping and hyperparameter optimization using Optuna. Optuna was used to optimize the learning rate, weight decay and dropout. The best hyperparameters were then used to train the final tuned EfficientNet-B0 model.

## 📊 Final Model Performance

| Metric | Score |
|---|---:|
| Test Accuracy | **96.54%** |
| Precision | **96.56%** |
| Recall | **96.53%** |
| F1 Score | **96.53%** |

The final model achieved approximately **96.54% test accuracy** with strong precision, recall and F1 performance.

## 🌱 Dataset

The project uses the **PlantVillage dataset** for plant disease classification. The dataset contains images of healthy and diseased plant leaves from multiple plant species. The model was trained to classify images into 38 different disease and healthy classes.

The dataset is not included in this repository because of its large size.

## 🏷️ Supported Classes

The model supports 38 classes covering:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper Bell
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

Both healthy and diseased conditions are included where available.

## ⚙️ Hyperparameter Optimization

**Optuna** was used for automated hyperparameter optimization. The main parameters searched were:

- Learning Rate
- Weight Decay
- Dropout

The best configuration was selected based on validation performance and then used to train the final tuned EfficientNet-B0 model.

## 🧪 Evaluation

The final model was evaluated using multiple metrics instead of relying only on accuracy. Evaluation included Accuracy, Precision, Recall, F1 Score, Confusion Matrix and Classification Report. Top-3 predictions were also implemented to provide additional information about the model's predictions.

## 🖥️ Streamlit Application

PlantCare AI provides a Streamlit-based interface where users can upload a plant leaf image and click **Detect Disease**. The application then preprocesses the image, performs inference using EfficientNet-B0 and displays the predicted disease, confidence score and Top-3 predictions.

### Application Workflow

Upload Image → Detect Disease → Model Inference → Prediction → Confidence Score → Top-3 Predictions

## 🎯 Example Prediction

Example output:

**Prediction:** Apple — Black rot

**Confidence:** 98.60%

**Top 3 Predictions:**

1. Apple — Black rot
2. Apple — Cedar apple rust
3. Cherry — Healthy

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- Optuna
- Scikit-learn
- NumPy
- Pillow
- Streamlit

## 📁 Project Structure

```text
plant-disease-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── Notebook/
│   ├── Data_Loading.ipynb
│   ├── Custom_CNN.ipynb
│   ├── Training.ipynb
│   ├── Evaluation.ipynb
│   └── Inference.ipynb
│
├── Models/
│   └── best_tuned_efficientnet.pth
│
└── Data/
    └── plantvillage/


    👨‍💻 Author

Shivendra Pratap Singh

Computer Science & Engineering | AI/ML

GitHub: https://github.com/Shivendra1230

⭐ Project Highlights

EfficientNet-B0 • Transfer Learning • Optuna • Early Stopping • Multi-Metric Evaluation • Top-3 Predictions • Streamlit • GPU Support