import joblib
import pandas as pd
from typing import Dict

# Load model and artifacts
model = joblib.load("data/model/model.pkl")
encoders = joblib.load("data/model/encoders.pkl")
feature_columns = joblib.load("data/model/feature_columns.pkl")


# --------------------------------
# Safe Label Encoding
# --------------------------------
# Encodes a single categorical value using a fitted LabelEncoder.
# Returns the encoded label if known; otherwise, returns -1 to handle unseen or problematic
# categories without causing transformation errors.

def safe_encode_column(val, encoder):
    try:
        val_str = str(val)
        if val_str in encoder.classes_:
            return encoder.transform([val_str])[0]
        else:
            return -1
    except Exception:
        return -1


# --------------------------------
# Prepare Input for Prediction
# --------------------------------
# Constructs a dictionary representing a single flight record, formatted to match the model's
# expected input structure for prediction.

def prepare_input(raw_input: Dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    # Apply encoders per column
    for col in encoders:
        if col in df.columns:
            encoder = encoders[col]
            df[col] = df[col].apply(lambda x: safe_encode_column(x, encoder))

    # Fill missing columns the model expects
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_columns]

    # Convert to numeric and replace remaining NaNs
    df = df.apply(pd.to_numeric, errors='coerce').fillna(-1)

    # Final sanity check
    assert not df.isnull().any().any(), "❌ Still has NaNs — check input or encoders."

    return df


# --------------------------------
# Predict Function
# --------------------------------
# Accepts raw input features and outputs the predicted likelihood of a flight delay using the
# trained model.

def predict_delay(raw_input: Dict) -> Dict:
    X = prepare_input(raw_input)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "predicted_class": int(prediction),
        "delay_probability": round(float(probability), 4)
    }


# Example usage
if __name__ == "__main__":
    test_input = {
        "DAY_OF_MONTH": 24,
        "DAY_OF_WEEK": 5,
        "OP_UNIQUE_CARRIER": "AA",
        "TAIL_NUM": "N123AA",
        "OP_CARRIER_FL_NUM": 1533,
        "ORIGIN": "ORD",
        "DEST": "DFW",
        "DEP_DEL15": 1.0,
        "DEP_TIME_BLK": "0900-0959",
        "DISTANCE": 1013.0,
        "ARR_TIME_BLK": "1200-1259",
        "ORIGIN_SEQ_ID": "04",
        "DEST_SEQ_ID": "02"
    }

    result = predict_delay(test_input)
    print("Prediction Result:", result)