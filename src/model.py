# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def run_ml_model(cleaned_csv_path="data/cleaned/cleaned_uber_lyft.csv"):
    # Load cleaned data
    df = pd.read_csv(cleaned_csv_path)

    # -----------------------------
    # 1. Select features for the model
    # -----------------------------
    features = ["distance", "surge_multiplier", "hour", "weekday", "visibility", "temperature"]
    target = "price"

    X = df[features]
    y = df[target]

    # -----------------------------
    # 2. Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. Scaling (optional)
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # 4. Linear Regression Model
    # -----------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # -----------------------------
    # 5. Random Forest Regressor (optional)
    # -----------------------------
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # -----------------------------
    # 6. Evaluation
    # -----------------------------
    print("\n--- Linear Regression Metrics ---")
    print(f"RMSE: {mean_squared_error(y_test, y_pred_lr) ** 0.5:.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}")
    print(f"R²: {r2_score(y_test, y_pred_lr):.2f}")

    print("\n--- Random Forest Metrics ---")
    print(f"RMSE: {mean_squared_error(y_test, y_pred_rf) ** 0.5:.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"R²: {r2_score(y_test, y_pred_rf):.2f}")

    # -----------------------------
    # 7. Save models and scaler
    # -----------------------------
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(lr_model, "outputs/models/linear_regression_model.pkl")
    joblib.dump(rf_model, "outputs/models/random_forest_model.pkl")
    joblib.dump(scaler, "outputs/models/scaler.pkl")

    print("\n[INFO] Models and scaler saved in outputs/models/")

    return lr_model, rf_model, scaler
