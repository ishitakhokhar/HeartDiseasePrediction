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

# --------------------------------------------------
# APP INITIALIZATION
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# LOAD MODEL & SCALER (LOAD ONCE)
# --------------------------------------------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# --------------------------------------------------
# FEATURE COLUMNS (MUST MATCH TRAINING)
# --------------------------------------------------
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

# --------------------------------------------------
# PRECOMPUTE DASHBOARD DATA (RUNS ONCE)
# --------------------------------------------------
try:
    df = pd.read_csv("cardio_train.csv", sep=";")

    y_true = df["cardio"]
    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Summary stats
    total_records = len(y_pred)
    high_risk = int(np.sum(y_pred == 1))
    low_risk = int(np.sum(y_pred == 0))
    disease_data = [low_risk, high_risk]
    risk_percentage = round((high_risk / total_records) * 100, 2)

    # Metrics
    accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
    precision = round(precision_score(y_true, y_pred) * 100, 2)
    recall = round(recall_score(y_true, y_pred) * 100, 2)
    f1 = round(f1_score(y_true, y_pred) * 100, 2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_data = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1])
    }

    # Feature importance (Random Forest)
    feature_names = FEATURE_COLUMNS
    feature_values = model.feature_importances_.tolist()

except Exception as e:
    print("Dashboard precompute failed:", e)

    # Fallback values (app will still run)
    disease_data = []
    feature_names = []
    feature_values = []
    total_records = high_risk = low_risk = 0
    risk_percentage = accuracy = precision = recall = f1 = 0
    cm_data = {}

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

# HOME / DASHBOARD (FAST)
@app.route("/")
def index():
    return render_template(
        "index.html",
        disease_data=disease_data,
        risk_labels=feature_names,
        risk_values=feature_values,
        total_records=total_records,
        high_risk=high_risk,
        low_risk=low_risk,
        risk_percentage=risk_percentage,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cm_data=cm_data
    )

# DETECT PAGE
@app.route("/detect")
def detect():
    return render_template("detect.html")

# USER PREDICTION
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

        prob_disease = model.predict_proba(data_scaled)[0][1]
        risk_percentage = round(prob_disease * 100, 2)

        # Risk label
        if risk_percentage < 30:
            risk_label = "Low Risk"
        elif risk_percentage < 60:
            risk_label = "Moderate Risk"
        else:
            risk_label = "High Risk"

        # Final decision
        result_text = (
            "Heart Disease Detected"
            if prob_disease >= 0.5
            else "No Risk of Heart Disease"
        )

        return render_template(
            "result.html",
            result=result_text,
            risk=risk_label,
            risk_percentage=risk_percentage
        )

    except Exception as e:
        return render_template("error.html", error=str(e))

# INSIGHTS PAGE
@app.route("/insights")
def insights():
    return render_template("insights.html")

# ABOUT PAGE
@app.route("/about")
def about():
    return render_template("about.html")

# --------------------------------------------------
# LOCAL RUN (NOT USED BY RENDER)
# --------------------------------------------------
if __name__ == "__main__":
    app.run()
