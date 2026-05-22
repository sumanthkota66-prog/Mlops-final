import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model():
    print("Fetching and preprocessing Car CO2 data...")

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "Car_CO2.csv"
    MODEL_PATH = BASE_DIR / "model.pkl"
    THRESHOLD_FILE = BASE_DIR / "best_r2.txt"

    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    required_columns = ["volume", "weight", "cylinders", "fuel_type", "co2"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    df = df.fillna(df.mean(numeric_only=True))

    X = df[["volume", "weight", "cylinders", "fuel_type"]]
    y = df["co2"]

    numeric_features = ["volume", "weight", "cylinders"]
    categorical_features = ["fuel_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    mlflow.set_experiment("Car_CO2_Volume_Prediction")

    with mlflow.start_run():
        print("Training Car CO2 regression pipeline...")

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(pipeline, "model")

        if os.path.exists(THRESHOLD_FILE):
            with open(THRESHOLD_FILE, "r") as f:
                historical_best = float(f.read().strip())
        else:
            historical_best = 0.0

        print("\n--- CHAMPION VS CHALLENGER ---")
        print(f"Current R2 Score: {r2}")
        print(f"Best R2 Score: {historical_best}")
        print(f"Mean Squared Error: {mse}")

        if r2 > historical_best:
            print("New best model found. Saving model...")

            with open(MODEL_PATH, "wb") as f:
                pickle.dump(pipeline, f)

            with open(THRESHOLD_FILE, "w") as f:
                f.write(str(r2))

            print("Model saved successfully.")
        else:
            print("Model not better than previous best. Skipping save.")


if __name__ == "__main__":
    train_model()
