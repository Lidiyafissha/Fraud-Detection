# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocess_fraud(df: pd.DataFrame) -> pd.DataFrame:
    # ==============================
    # BASIC VALIDATION
    # ==============================
    required_columns = {
        "user_id",
        "purchase_value",
        "age",
        "ip_address",
        "class",
        "signup_time",
        "purchase_time",
        "device_id",
        "country"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required fraud columns: {missing}")

    if df.empty:
        raise ValueError("Fraud dataset is empty")

    # ==============================
    # HANDLE MISSING VALUES
    # ==============================
    df = df.copy()

    numeric_cols = ["purchase_value", "age"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["device_id", "country"]
    for col in categorical_cols:
        df[col] = df[col].fillna("UNKNOWN")

    # Target validation
    if not set(df["class"].unique()).issubset({0, 1}):
        raise ValueError("Target column `class` must be binary (0/1)")

    return df

    
