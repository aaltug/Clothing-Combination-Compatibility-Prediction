# 👗 Outfit Compatibility Prediction Project

This project aims to predict the compatibility of 3-piece fashion outfits consisting of a top, bottom, and shoes using image-based features and machine learning. By extracting features from outfit images and feeding them into a trained model, users can receive a compatibility score for their selected combination.

---

## 🧠 Technologies & Methods

- **Image Feature Extraction**: [VGG19](https://arxiv.org/abs/1409.1556) pre-trained CNN model
- **Category Detection**: HuggingFace-based classifier for clothing item categorization
- **Compatibility Classifier**:
  - XGBoost (main model)
  - MLPClassifier (tested as alternative)
- **User Interface**: Basic Python GUI (can be developed using Streamlit or Tkinter)
- **Dataset**: Polyvore Outfits (filtered to 3-piece combinations)

---

# ✅ Requirements
-Python 3.7+
-Libraries: torch, transformers, xgboost, scikit-learn, Pillow, etc.
-Pre-trained models for feature extraction and classification

---

## 📸 Example

<img width="1121" height="757" alt="outfit28" src="https://github.com/user-attachments/assets/e1ff9ff8-1368-4c02-acd7-00c36dd6a916" />







<img width="1135" height="777" alt="--" src="https://github.com/user-attachments/assets/4b743d18-8900-403d-bfa0-5b49f7d36bfe" />
