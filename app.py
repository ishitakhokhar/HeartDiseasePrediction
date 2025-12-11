import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)


model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        user_input = [float(x) for x in request.form.values()]

        
        data = np.array(user_input).reshape(1, -1)

        
        data_scaled = scaler.transform(data)

        
        prediction = model.predict(data_scaled)[0]

  
        if prediction == 1:
            result = " High Risk of Heart Disease"
        else:
            result = " No Heart Disease Detected"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
