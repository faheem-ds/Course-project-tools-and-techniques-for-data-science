import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def summarize_numeric(df: pd.DataFrame):
    """Print summary statistics of numeric columns."""
    print("\n[EDA] Numeric summary:")
    print(df.describe())

def summarize_categorical(df: pd.DataFrame):
    """Print counts for categorical columns."""
    cat_cols = df.select_dtypes(include='object').columns
    print("\n[EDA] Categorical value counts:")
    for col in cat_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())

def plot_missing_values(df: pd.DataFrame, save_path="outputs/figures/missing_values.png"):
    """Visualize missing values heatmap."""
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[EDA] Missing values heatmap saved to {save_path}")

def plot_numeric_distributions(df: pd.DataFrame, numeric_cols=None, save_dir="outputs/figures"):
    """Plot histograms for numeric columns."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    os.makedirs(save_dir, exist_ok=True)
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        file_path = os.path.join(save_dir, f"{col}_distribution.png")
        plt.savefig(file_path)
        plt.close()
        print(f"[EDA] Distribution plot for '{col}' saved to {file_path}")

def correlation_heatmap(df: pd.DataFrame, numeric_cols=None, save_path="outputs/figures/correlation_heatmap.png"):
    """Plot correlation heatmap of numeric features."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[EDA] Correlation heatmap saved to {save_path}")

def time_based_analysis(df: pd.DataFrame, time_col='pickup_time', save_dir="outputs/figures"):
    """Generate plots by hour, day of week."""
    if time_col not in df.columns:
        print(f"[EDA] Column '{time_col}' not found. Skipping time-based analysis.")
        return
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.day_name()
    
    # Trips by hour
    hourly = df.groupby('hour').size()
    plt.figure(figsize=(10,4))
    sns.lineplot(x=hourly.index, y=hourly.values)
    plt.title("Trips by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Number of Trips")
    plt.savefig(os.path.join(save_dir, "trips_by_hour.png"), bbox_inches='tight')
    plt.close()
    
    # Trips by day of week
    daily = df.groupby('day_of_week').size()
    days_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    daily = daily.reindex(days_order)
    plt.figure(figsize=(10,4))
    sns.barplot(x=daily.index, y=daily.values)
    plt.title("Trips by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Number of Trips")
    plt.savefig(os.path.join(save_dir, "trips_by_day.png"), bbox_inches='tight')
    plt.close()
    
    print(f"[EDA] Time-based analysis plots saved to {save_dir}")


def run_eda(df: pd.DataFrame):
    """
    Runs all EDA steps:
    - Numeric & categorical summary
    - Missing values heatmap
    - Numeric distributions
    - Correlation heatmap
    - Time-based analysis
    """
    summarize_numeric(df)
    summarize_categorical(df)
    plot_missing_values(df)
    plot_numeric_distributions(df)
    correlation_heatmap(df)
    time_based_analysis(df)
