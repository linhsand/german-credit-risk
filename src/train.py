# src/train.py
from preprocess import load_and_clean_data, get_feature_types, build_preprocessor,feature_engineering
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def main():
    df = load_and_clean_data("data/german_credit_data.csv")
    df=feature_engineering(df,do_poly=False,drop_orig=False)

    numeric_cols,categorical_cols=get_feature_types(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols,scaler="robust")


    numeric_cols, categorical_cols = get_feature_types(df)
  

    if "target" not in df.columns and "Risk" in df.columns:
        df["target"]=df["Risk"].map({"good":0,"bad":1})


    X = df.drop(columns=[c for c in ["Risk","target"] if c in df.columns])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print("DEBUG: type(preprocessor) =", type(preprocessor))
    print("DEBUG: preprocessor repr:", repr(preprocessor)[:300])
    print("DEBUG: pipeline.named_steps:")
    for n, s in pipeline.named_steps.items():
        print(" -", n, type(s))

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model.joblib")
    print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()

    
