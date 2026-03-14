import joblib
import pandas as pd
from flask import Flask, render_template, request
from config.paths import (
    MODEL_FILE,
    SCALER_FILE,
    SELECTED_FEATURES_FILE,
    ENCODER_FILE,
    TARGET_ENCODER_FILE,
)

app = Flask(__name__)

# Load model, scaler, and metadata
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
selected_features = joblib.load(SELECTED_FEATURES_FILE)
encoders = joblib.load(ENCODER_FILE)
target_encoder = joblib.load(TARGET_ENCODER_FILE)


@app.route("/")
def index():
    return render_template(
        "index.html",
        prediction=None,
        features=selected_features,
        encoders=encoders,
        form_data={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form dynamically based on selected features
        input_data = {}
        original_form = {}
        for feature in selected_features:
            val = request.form.get(feature)
            original_form[feature] = val
            if val is None:
                raise ValueError(f"Missing required field: {feature}")

            # Check if feature needs label encoding
            if feature in encoders:
                le = encoders[feature]
                # If val is a number string but the encoder expects labels like 'Yes'/'No'
                # this happens if the browser caches the old form.
                if val not in le.classes_:
                    # Try to see if it was an index from an old form version
                    try:
                        idx = int(val)
                        if 0 <= idx < len(le.classes_):
                            input_data[feature] = idx
                        else:
                            raise ValueError(
                                f"Value '{val}' not found in labels for {feature}: {le.classes_}"
                            )
                    except ValueError:
                        raise ValueError(
                            f"Value '{val}' not found in labels for {feature}: {le.classes_}"
                        )
                else:
                    input_data[feature] = le.transform([val])[0]
            else:
                input_data[feature] = float(val)

        # Create input DataFrame in the correct order
        features_df = pd.DataFrame([input_data])[selected_features]

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Decode prediction using target_encoder
        prediction_label = target_encoder.inverse_transform([prediction])[0]

        return render_template(
            "index.html",
            prediction=prediction_label,
            features=selected_features,
            encoders=encoders,
            form_data=original_form,
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
