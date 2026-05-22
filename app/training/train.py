import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_tracking_uri("http://127.0.0.1:5555")

# Load dataset
df = pd.read_csv("Car_CO2.csv")

# Features (X) and Target (y)
X = df[["volume", "weight", "cylinders", "fuel_type"]]
y = df["co2"]

categorical_features = ["fuel_type"]
numeric_features = ["volume", "weight", "cylinders"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow Run
with mlflow.start_run():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    result = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    mlflow.register_model(
        model_uri=result.model_uri,
        name="car-co2-volume-regression-model"
    )

    print(f"Model logged with R2: {r2}")

# Save model to a .pkl file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl!")
