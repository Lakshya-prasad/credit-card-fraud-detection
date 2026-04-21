import pandas as pd

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