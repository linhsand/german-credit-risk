# German Credit Risk Prediction (ML Project)

This project builds a machine learning model to predict **credit risk** (good/bad) using the German Credit dataset.  
Project includes full pipeline: **EDA → Preprocessing → Feature Engineering → Model Training → Saving Model**.

---

## 📁 Project Structure
german-credit-ml/
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

## 🔧 Installation
### 1. Create virtual environment
bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install -r requirements.txt

jupyter notebook notebooks/EDA.ipynb
* EDA includes:
Checking missing values
Distribution plots
Target balance
Correlation
Outlier detection

* Preprocessing
File: src/preprocess.py
Includes:
Remove unnamed columns
Convert dtypes
Basic cleaning
Automated feature type detection
Feature engineering
Scaling (Standard / MinMax / Robust / PowerTransform)
One-hot encoding
ColumnTransformer pipeline

* Model Training
Run training script:
python src/train.py
Pipeline:
Feature engineering
Preprocessor (num + cat pipelines)
RandomForestClassifier
Train-test split
Evaluation (accuracy + classification report)
Save model to models/model.joblib
Sample accuracy achieved: 0.73–0.75

* Model Output
After training, model is saved to:
models/model.joblib
Load model for inference:
import joblib
model = joblib.load("models/model.joblib")
pred = model.predict(df_new)
