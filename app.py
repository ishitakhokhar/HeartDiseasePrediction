import pickle
import pandas as pd
import numpy as np

from flask import Flask, render_template, request
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

app = Flask(__name__)

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# IMPORTANT:
# These MUST match EXACTLY the columns used during training
FEATURE_COLUMNS = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

# -------------------------------
# HOME / DASHBOARD
# -------------------------------
@app.route("/")
def index():

    # Load dataset
    df = pd.read_csv("cardio_train.csv", sep=";")

    # Target column
    y_true = df["cardio"]

    # Features (EXACT SAME as training)
    X = df[FEATURE_COLUMNS]

    # Scale features
    X_scaled = scaler.transform(X)

    # -------------------------------
    # MODEL PREDICTIONS
    # -------------------------------
    y_pred = model.predict(X_scaled)

    # Prediction summary
    total_records = len(y_pred)
    high_risk = int(np.sum(y_pred == 1))
    low_risk = int(np.sum(y_pred == 0))

    disease_data = [low_risk, high_risk]

    risk_percentage = round((high_risk / total_records) * 100, 2)

    # -------------------------------
    # MODEL METRICS
    # -------------------------------
    accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
    precision = round(precision_score(y_true, y_pred) * 100, 2)
    recall = round(recall_score(y_true, y_pred) * 100, 2)
    f1 = round(f1_score(y_true, y_pred) * 100, 2)

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    cm = confusion_matrix(y_true, y_pred)

    cm_data = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1])
    }

    # -------------------------------
    # FEATURE IMPORTANCE (Random Forest)
    # -------------------------------
    feature_names = FEATURE_COLUMNS
    feature_values = model.feature_importances_.tolist()

    return render_template(
        "index.html",

        # Charts
        disease_data=disease_data,
        risk_labels=feature_names,
        risk_values=feature_values,

        # Summary cards
        total_records=total_records,
        high_risk=high_risk,
        low_risk=low_risk,
        risk_percentage=risk_percentage,

        # Metrics
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,

        # Confusion Matrix
        cm_data=cm_data
    )

# -------------------------------
# DETECT PAGE
# -------------------------------
@app.route("/detect")
def detect():
    return render_template("detect.html")

# -------------------------------
# PREDICTION (USER INPUT)
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = [
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["height"]),
            float(request.form["weight"]),
            float(request.form["ap_hi"]),
            float(request.form["ap_lo"]),
            float(request.form["cholesterol"]),
            float(request.form["gluc"]),
            float(request.form["smoke"]),
            float(request.form["alco"]),
            float(request.form["active"])
        ]

        data = np.array(user_input).reshape(1, -1)
        data_scaled = scaler.transform(data)

        # Always use probability of disease (class 1)
        prob_disease = model.predict_proba(data_scaled)[0][1]
        risk_percentage = round(prob_disease * 100, 2)

        # 🔹 RISK LEVEL LOGIC (THIS GOES HERE)
        if risk_percentage < 30:
            risk_label = "Low Risk"
        elif risk_percentage < 60:
            risk_label = "Moderate Risk"
        else:
            risk_label = "High Risk"

        # 🔹 FINAL DECISION (threshold logic)
        threshold = 0.5
        if prob_disease >= threshold:
            result_text = "Heart Disease Detected"
        else:
            result_text = "No Risk of Heart Disease"

        return render_template(
            "result.html",
            result=result_text,
            risk=risk_label,
            risk_percentage=risk_percentage
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------
# INSIGHTS
# -------------------------------
@app.route("/insights")
def insights():
    return render_template("insights.html")

# -------------------------------
# ABOUT
# -------------------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run()
