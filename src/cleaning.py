# wrangle.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_missing_heatmap(df):
    os.makedirs("outputs/figures/wrangling", exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap (Before Cleaning)")
    plt.tight_layout()
    plt.savefig("outputs/figures/wrangling/missing_values_before_cleaning.png")
    plt.close()


def clean_data(df):
    print("\n[WRANGLING] Starting data cleaning...")

    # ----------------------------------------------------
    # 0. Save heatmap BEFORE cleaning
    # ----------------------------------------------------
    save_missing_heatmap(df)

    # Summary BEFORE cleaning
    print("\n[SUMMARY BEFORE CLEANING]")
    print(df.info())
    print(df.isnull().sum())
    print(f"Duplicates before: {df.duplicated().sum()}")
    original_rows = df.shape[0]

    # ----------------------------------------------------
    # 1. Remove duplicates
    # ----------------------------------------------------
    df = df.drop_duplicates()
    print(f"\n[INFO] Removed duplicates: {original_rows - df.shape[0]}")

    # ----------------------------------------------------
    # 2. Fix data types
    # ----------------------------------------------------
    print("\n[INFO] Fixing data types...")

    num_cols = ["price", "distance", "surge_multiplier",
                "temperature", "humidity", "visibility", "latitude", "longitude"]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["timestamp", "datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ----------------------------------------------------
    # 3. Handle missing values
    # ----------------------------------------------------
    print("\n[INFO] Handling missing values...")

    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")

    # Numeric fill
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical fill
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"Missing values after: {missing_after}")

    # ----------------------------------------------------
    # 4. Outlier Removal (IQR method)
    # ----------------------------------------------------
    print("\n[INFO] Removing outliers from 'price' (IQR method)...")
    total_outliers_removed = 0
    if 'price' in df.columns:
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = df.shape[0]
        df = df[(df['price'] >= lower) & (df['price'] <= upper)]
        removed = before - df.shape[0]
        total_outliers_removed = removed
        if removed > 0:
            print(f"[INFO] Outliers removed from price: {removed}")
    print(f"\nTotal outliers removed: {total_outliers_removed}")

    # ----------------------------------------------------
    # 5. Feature Engineering
    # ----------------------------------------------------
    print("\n[INFO] Creating time-based features...")

    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["weekday"] = df["timestamp"].dt.weekday

    print("[INFO] Feature engineering complete.")

    # ----------------------------------------------------
    # 6. Save cleaned dataset
    # ----------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_uber_lyft.csv", index=False)
    print("\n[SAVED] Cleaned dataset saved at: data/processed/cleaned_uber_lyft.csv")

    return df


def wrangle_data(df):
    return clean_data(df)
