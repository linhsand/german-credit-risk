# German Credit Risk Prediction

This project builds a machine learning model to predict whether a loan applicant is a **good or bad credit risk** using the German Credit dataset.

The goal is to apply a full machine learning workflow including **data preprocessing, feature engineering, model training, and evaluation**.

---

# Dataset

Dataset: German Credit Data

Source:
https://www.kaggle.com/datasets/uciml/german-credit

The dataset contains financial and personal information about loan applicants.

Examples of features:

* Age
* Credit amount
* Duration
* Employment status
* Housing
* Savings account

Target variable:

| Value | Meaning          |
| ----- | ---------------- |
| 0     | Good credit risk |
| 1     | Bad credit risk  |

---

# Project Structure

```
german-credit-risk/
│
├── data/
│   └── german_credit_data.csv
│
├── src/
│   ├── preprocess.py
│   └── train.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── models/
│   └── model.joblib
│
├── requirements.txt
└── README.md
```

---

# Machine Learning Workflow

1. Load and clean dataset
2. Handle missing values
3. Feature engineering
4. Train / test split
5. Train machine learning model
6. Evaluate performance

---

# Model

Algorithm used:

**Random Forest Classifier**

Libraries:

* Scikit-learn
* Pandas
* NumPy

---

# Model Performance

Test results:

Accuracy: **0.71**

ROC-AUC: **0.78**

Classification report:

| Class       | Precision | Recall | F1-score |
| ----------- | --------- | ------ | -------- |
| Good Credit | 0.83      | 0.74   | 0.78     |
| Bad Credit  | 0.52      | 0.65   | 0.58     |

---

# Tech Stack

Python
Pandas
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook

---

# How to Run

Clone the repository:

```
git clone https://github.com/linhsand/german-credit-risk.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python src/train.py
```

The trained model will be saved to:

```
models/model.joblib
```

---

# Future Improvements

* Hyperparameter tuning
* Model comparison (Logistic Regression, XGBoost)
* Feature importance analysis
* Deployment as API

---

# Author

Cát Linh
