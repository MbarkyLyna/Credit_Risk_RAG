import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import shap
from src.rag.embeddings import build_vectorstore

FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]

def train_and_save():
    print("Training model...")
    df = pd.read_csv("data/cs-training.csv", index_col=0)
    df = df.dropna(subset=['SeriousDlqin2yrs'])
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
    df = df[df['age'] > 18]
    df = df[df['age'] < 100]
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1]

    X = df[FEATURES]
    y = df['SeriousDlqin2yrs']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=10, eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)
    os.makedirs("src/model", exist_ok=True)
    joblib.dump(model, "src/model/credit_model.pkl")
    print("Model saved.")

if __name__ == "__main__":
    if not os.path.exists("src/model/credit_model.pkl"):
        train_and_save()
    if not os.path.exists("chroma_db"):
        print("Building vectorstore...")
        build_vectorstore()
        print("Vectorstore built.")