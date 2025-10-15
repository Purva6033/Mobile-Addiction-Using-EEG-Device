import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# =============================
# Custom Transformer for DL reshaping
# =============================
class DLReshaper:
    def __init__(self, mode='cnn'):
        self.mode = mode

    def transform(self, X):
        if self.mode == 'cnn':
            return X.reshape((X.shape[0], X.shape[1], 1))
        elif self.mode == 'rnn':
            return X.reshape((X.shape[0], 1, X.shape[1]))
        else:
            raise ValueError("mode must be 'cnn' or 'rnn'")

# =============================
# Load saved models
# =============================
scaler = joblib.load("models/scaler.pkl")
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
rnn_model = tf.keras.models.load_model("models/rnn_model.h5")
rf_model = joblib.load("models/random_forest_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
meta_model = joblib.load("models/hybrid_model.pkl")  # Ensure correct path

# =============================
# Class mapping
# =============================
class_map = {0: "Normal", 1: "Mild Addiction", 2: "Severe Addiction"}

# =============================
# Prediction function
# =============================
def predict_eeg(input_data, output_file="predictions.csv"):
    """
    input_data: str (CSV path) or np.ndarray (single or multiple samples)
    """
    # Load data
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"{input_data} not found.")
        data = pd.read_csv(input_data)
        X = data.values
        # Keep only first 988 features if more
        if X.shape[1] > 988:
            X = X[:, :988]
        elif X.shape[1] < 988:
            raise ValueError("Input data must have 988 features per sample.")
    elif isinstance(input_data, np.ndarray):
        X = input_data
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 988:
            raise ValueError("Input array must have 988 features per sample.")
        data = pd.DataFrame(X)
    else:
        raise TypeError("input_data must be a CSV file path or a NumPy array.")

    # Preprocessing
    X_scaled = scaler.transform(X)
    X_cnn = DLReshaper('cnn').transform(X_scaled)
    X_rnn = DLReshaper('rnn').transform(X_scaled)

    # Base model predictions
    cnn_preds = cnn_model.predict(X_cnn)
    lstm_preds = lstm_model.predict(X_rnn)
    rnn_preds = rnn_model.predict(X_rnn)
    rf_preds = rf_model.predict_proba(X_scaled)
    svm_preds = svm_model.predict_proba(X_scaled)

    # Stack features for meta model
    stacked_features = np.hstack([cnn_preds, lstm_preds, rnn_preds, rf_preds, svm_preds])

    # Hybrid prediction
    hybrid_preds = meta_model.predict(stacked_features)
    mapped_preds = [class_map[p] for p in hybrid_preds]

    # Add predictions to dataframe
    data['Predicted_Addiction_Level'] = mapped_preds

    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Print results
    for i, pred in enumerate(mapped_preds):
        print(f"Sample {i+1}: Predicted Addiction Level = {pred}")

# =============================
# Run prediction from CSV
# =============================
if __name__ == "__main__":
    predict_eeg("balanced_dataset_EEG.csv")
