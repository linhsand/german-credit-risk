# CREDIT RISK CLASSIFICATION (GERMAN CREDIT DATA)

A machine learning project to predict credit risk ("good" vs "bad") using the German Credit dataset.  
Includes: EDA, preprocessing, feature engineering, model training, and evaluation.

---

## 📁 PROJECT STRUCTURE
project/
│
├── data/
│ └── german_credit_data.csv
│
├── notebooks/
│ └── EDA.ipynb
│
├── src/
│ ├── preprocess.py
│ └── train.py
│
├── models/
│ └── model.joblib
│
└── README.md


---

## 🚀 FEATURES
- Full EDA with graphs & data insights  
- Automatic cleaning + encoding + scaling  
- Feature engineering (ratios, interactions, log transform, bins, flags, etc.)  
- Multiple scalers (Standard / MinMax / Robust / Power)  
- Sklearn Pipeline + ColumnTransformer  
- RandomForest baseline (accuracy: ~0.73–0.75)  
- Model saved as `.joblib`

---

## 🔧 INSTALLATION


pip install -r requirements.txt


---

## 🧹 PREPROCESSING (IN preprocess.py)
- Remove `Unnamed` index columns  
- Convert numeric fields  
- Generate new features  
- Handle missing values  
- Auto-detect numeric & categorical columns  
- Apply OneHot + scaling via ColumnTransformer  

---

## 🎯 TRAINING (train.py)


python src/train.py

Outputs:
- Accuracy & classification report  
- Saved model → `models/model.joblib`

---

## 📊 EVALUATION
You can evaluate inside the `EDA.ipynb` or create a separate evaluation script if needed.

---

## 📦 MODEL EXPORT
Model is saved automatically as:


models/model.joblib


---

## 🌐 GITHUB USAGE
To push this project:


git init
git add .
git commit -m "Initial ML credit risk project"
git branch -M main
git remote add origin https://github.com/
<your-username>/<repo-name>.git
git push -u origin main


---

## 🏁 SUMMARY
This project demonstrates a complete ML workflow:
- Data exploration  
- Cleaning & preprocessing  
- Feature engineering  
- ML pipeline training  
- Model exporting  
- Reproducible project structure
