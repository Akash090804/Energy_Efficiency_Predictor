from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your multi-output model and scaler
model = load_model("psobilstm_model.keras")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from form as floats
        inputs = [float(request.form[f'X{i}']) for i in range(1, 9)]
        input_array = np.array([inputs])  # Shape: (1, 8)

        # Scale input
        scaled_input = scaler.transform(input_array)  # Shape: (1, 8)

        # Reshape to (batch_size, timesteps=1, features=8)
        scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))

        # Predict: model returns a list of outputs for multi-output model
        predictions = model.predict(scaled_input)

        # predictions is a list: [y1_preds, y2_preds]
        heating_load = predictions[0][0][0]  # First sample, first output (y1)
        cooling_load = predictions[1][0][0]  # First sample, second output (y2)

        # Render results on the page
        return render_template('index.html',
                               heating=round(float(heating_load), 2),
                               cooling=round(float(cooling_load), 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
