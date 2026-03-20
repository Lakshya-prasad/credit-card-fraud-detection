# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)

    print("✅ Data Loaded Successfully")
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())

    return df


def preprocess_data(df):
    X = df.drop(columns="Class", axis=1)
    y = df["Class"]
    return X, y


def train_model():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=2
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    X_test_prediction = model.predict(X_test)

    # Accuracy
    test_accuracy = accuracy_score(y_test, X_test_prediction)

    print("\n📊 Model Performance")
    print("Accuracy score on Test Data :", test_accuracy)

    # Classification Report
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test, X_test_prediction))

    # Save model
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved successfully!")


if __name__ == "__main__":
    train_model()