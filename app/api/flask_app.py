import os
import sys
from flask import Flask, request, jsonify

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.model.predict_model import predict_delay

app = Flask(__name__)

@app.route("/")
def index():
    return "✈️ Flight Delay Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read incoming JSON
        input_data = request.get_json()

        # Pass to the prediction pipeline
        result = predict_delay(input_data)

        # Return result
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)