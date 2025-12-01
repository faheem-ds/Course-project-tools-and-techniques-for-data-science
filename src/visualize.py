# visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.constants import CLEANED_DATA_PATH

def run_all_charts(cleaned_csv_path=CLEANED_DATA_PATH):
    """
    Generate all charts for Step 4 from the cleaned CSV
    """
    # Load cleaned dataset
    df = pd.read_csv(cleaned_csv_path)

    # Create output folder
    os.makedirs("outputs/figures/charts", exist_ok=True)

    sns.set(style="whitegrid")

    # -----------------------------
    # 1. Price Distribution
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
    plt.title("Distribution of Ride Prices")
    plt.xlabel("Price ($)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/figures/charts/price_distribution.png")
    plt.close()

    # -----------------------------
    # 2. Price vs Distance
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='distance', y='price', data=df, alpha=0.5)
    plt.title("Price vs Distance")
    plt.xlabel("Distance (miles)")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("outputs/figures/charts/price_vs_distance.png")
    plt.close()

    # -----------------------------
    # 3. Price vs Surge Multiplier
    # -----------------------------
    plt.figure(figsize=(8,6))
    sns.boxplot(x='surge_multiplier', y='price', data=df)
    plt.title("Price vs Surge Multiplier")
    plt.xlabel("Surge Multiplier")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("outputs/figures/charts/price_vs_surge.png")
    plt.close()

    # -----------------------------
    # 4. Price by Hour
    # -----------------------------
    plt.figure(figsize=(10,6))
    sns.boxplot(x='hour', y='price', data=df)
    plt.title("Price Distribution by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig("outputs/figures/charts/price_by_hour.png")
    plt.close()

    # -----------------------------
    # 5. Price by Cab Type
    # -----------------------------
    if 'cab_type' in df.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x='cab_type', y='price', data=df)
        plt.title("Price by Cab Type")
        plt.xlabel("Cab Type")
        plt.ylabel("Price ($)")
        plt.tight_layout()
        plt.savefig("outputs/figures/charts/price_by_cab_type.png")
        plt.close()

    # -----------------------------
    # 6. Correlation Heatmap
    # -----------------------------
    plt.figure(figsize=(12,8))
    top_features = ['price','distance','surge_multiplier','latitude','moonPhase','visibility']
    corr = df[top_features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap (Cleaned Data)")
    plt.tight_layout()
    plt.savefig("outputs/figures/charts/correlation_heatmap.png")
    plt.close()

    # -----------------------------
    # 7. Optional: Ride Count by Hour
    # -----------------------------
    plt.figure(figsize=(10,6))
    sns.countplot(x='hour', data=df)
    plt.title("Ride Count by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Rides")
    plt.tight_layout()
    plt.savefig("outputs/figures/charts/ride_count_by_hour.png")
    plt.close()

    print("[INFO] All charts generated and saved in outputs/figures/charts/")
