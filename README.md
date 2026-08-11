# 🌿 PlantCare AI

AI-powered plant disease detection system using
Transfer Learning with EfficientNet-B0.

## 🚀 Features

- Plant disease classification
- 38 disease/healthy classes
- EfficientNet-B0 transfer learning
- Hyperparameter tuning with Optuna
- Top-3 predictions
- Confidence scores
- Streamlit web application
- GPU accelerated inference

## 🧠 Model

Architecture: EfficientNet-B0

Test Accuracy: 96.54%

Precision: 96.56%

Recall: 96.53%

F1 Score: 96.53%

## 🔬 Training Pipeline

1. Dataset preparation
2. Data augmentation
3. Baseline CNN
4. Augmented CNN
5. Transfer Learning
6. EfficientNet-B0
7. Hyperparameter tuning using Optuna
8. Early stopping
9. Final evaluation
10. Streamlit deployment

## 🛠️ Tech Stack

Python
PyTorch
Torchvision
Optuna
Scikit-learn
Streamlit
Pillow

## ▶️ Run Locally

```bash
conda activate plant
pip install -r requirements.txt
python -m streamlit run app.py