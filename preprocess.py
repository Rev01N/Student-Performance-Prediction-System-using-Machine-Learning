"""
preprocess.py
-------------
Data cleaning, encoding, and normalisation pipeline.

Steps:
  1. Load raw CSV
  2. Report & handle missing values
  3. Label-encode Final_Result  (Pass=1, Fail=0)
  4. StandardScaler on numeric features
  5. Save cleaned CSV + scaler object
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

RAW_CSV     = os.path.join(DATA_DIR,   "student_data.csv")
CLEAN_CSV   = os.path.join(DATA_DIR,   "student_data_clean.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH= os.path.join(MODELS_DIR, "label_encoder.pkl")

FEATURE_COLS = ["Study_Hours", "Attendance", "Previous_Marks",
                "Assignments", "Internal_Marks"]
TARGET_COL   = "Final_Result"


def preprocess(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Run the full preprocessing pipeline. Returns (X_scaled_df, y_series)."""

    for d in [OUTPUT_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_CSV)
    if verbose:
        print("\n[Preprocess] Raw dataset loaded.")
        print(f"  Shape : {df.shape}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")

    # ── 2. Missing values ─────────────────────────────────────────────────────
    missing = df.isnull().sum()
    if verbose:
        print(f"\n[Preprocess] Missing values:\n{missing.to_string()}")

    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    if df[TARGET_COL].isnull().any():
        df[TARGET_COL].fillna(df[TARGET_COL].mode()[0], inplace=True)

    # ── 3. Encode target ──────────────────────────────────────────────────────
    le = LabelEncoder()
    df["Final_Result_Encoded"] = le.fit_transform(df[TARGET_COL])
    joblib.dump(le, ENCODER_PATH)
    if verbose:
        print(f"\n[Preprocess] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── 4. Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    X_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    joblib.dump(scaler, SCALER_PATH)
    if verbose:
        print(f"\n[Preprocess] Features scaled with StandardScaler.")

    # ── 5. Save clean CSV ─────────────────────────────────────────────────────
    clean_df = df.copy()
    for i, col in enumerate(FEATURE_COLS):
        clean_df[col] = X_scaled[:, i]
    clean_df.to_csv(CLEAN_CSV, index=False)
    if verbose:
        print(f"\n[Preprocess] Clean data saved → {CLEAN_CSV}")
        print(f"  Class distribution:\n{df[TARGET_COL].value_counts().to_string()}")
        print("\n[Preprocess] ✔ Complete.\n")

    y = df["Final_Result_Encoded"]
    return X_df, y


if __name__ == "__main__":
    preprocess()
