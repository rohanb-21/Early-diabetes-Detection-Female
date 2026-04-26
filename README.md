# 🩺 GlycoSense — Early Diabetes Screening for Women

An ML-powered web application that predicts early-stage diabetes 
risk in women using the Pima Indians Diabetes Database.

---

## 📊 Dataset
- **Source:** [Pima Indians Diabetes Database — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **768 female patients** · 9 clinical features
- **Target:** Diabetic (1) or Non-Diabetic (0)

---

## 🤖 4 Models Trained & Compared

| Model | AUC | Accuracy | F1 Score |
|---|---|---|---|
| 🟣 Gradient Boosting | 0.8296 | 75.97% | 0.6476 |
| 🔵 Random Forest | 0.8239 | 72.73% | 0.6441 |
| 🟢 Logistic Regression | 0.8202 | 74.03% | 0.6610 |
| 🟡 KNN Classifier | 0.7914 | 74.68% | 0.6061 |

---

## ⚙️ Key ML Concepts Used

- **Fake Zero Problem** — 5 columns had impossible zeros (e.g. Glucose=0, BMI=0). Replaced with NaN before training.
- **KNN Imputer (k=5)** — fills missing values using 5 nearest patients. Used inside ALL 4 pipelines.
- **KNN Classifier (k=11)** — one of the 4 prediction models. Distance-weighted voting.
- **Feature Engineering** — 5 new interaction features (Glucose×BMI, Age×Pregnancies, etc.)
- **Feature Dependency Analysis** — each feature dropped one at a time across all 4 models to measure AUC impact
- **Robust Scaler** — scales using median+IQR, resistant to lab outliers
- **5-Fold Cross Validation** — measures model consistency

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/rohanb-21/Early-diabetes-Detection-Female.git
cd Early-diabetes-Detection-Female
```

### 2. Create virtual environment
```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash
python3 train_model.py
```
This reads `diabetes.csv`, fixes fake zeros, engineers features,
trains all 4 models, runs feature dependency analysis and saves
4 `.pkl` files + `model_meta.json`.

### 5. Start the web app
```bash
python3 app.py
```

### 6. Open in browser
http://localhost:5000

---

## 📁 Project Structure

├── static/
│   └── index.html                  # Full medical web UI (4 tabs)
├── app.py                          # Flask REST API backend
├── train_model.py                  # Full ML pipeline
├── diabetes.csv                    # Pima Indians dataset
├── model_gradient_boosting.pkl     # Trained GB pipeline
├── model_random_forest.pkl         # Trained RF pipeline
├── model_logistic_regression.pkl   # Trained LR pipeline
├── model_knn_classifier.pkl        # Trained KNN pipeline
├── model_meta.json                 # Metrics + feature dependency data
├── requirements.txt                # Python dependencies
└── README.md                       # This file

---

## 🔬 Full ML Pipeline

diabetes.csv
↓
Fix Fake Zeros (0 → NaN in 5 columns)
↓
Feature Engineering (8 original + 5 new = 13 features)
↓
Train/Test Split (80% train, 20% test, stratified)
↓
Pipeline: KNN Imputer → Robust Scaler → Classifier
↓
Train 4 Models + 5-Fold Cross Validation
↓
Feature Dependency Analysis (52 retraining experiments)
↓
Save .pkl files → Flask API → Web UI

---

## 🌐 Web App Features

- **🔬 Predict Tab** — enter clinical values, choose model, get risk score
- **📊 Model Comparison Tab** — AUC, Accuracy, F1, CV scores for all 4 models
- **🧬 Feature Dependency Tab** — ranked table showing AUC drop per feature per model
- **⚙️ How It Works Tab** — full pipeline walkthrough + dataset details

---

## ⚠️ Disclaimer
For educational and screening purposes only.  
Not a substitute for professional medical advice.  
Always consult a licensed physician for clinical evaluation.