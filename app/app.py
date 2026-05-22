import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    volume = float(request.form["volume"])
    weight = float(request.form["weight"])
    cylinders = int(request.form["cylinders"])
    fuel_type = request.form["fuel_type"]

    feature_names = ["volume", "weight", "cylinders", "fuel_type"]
    features = pd.DataFrame([[volume, weight, cylinders, fuel_type]], columns=feature_names)

    prediction = model.predict(features)
    formatted_prediction = f"The predicted CO2 emission is {round(float(prediction[0]), 2)} g/km"

    return render_template("result.html", prediction=formatted_prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
