import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# =============================
# Custom reshaper
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
# Load models and scaler
# =============================
scaler = joblib.load("models/scaler.pkl")
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
rnn_model = tf.keras.models.load_model("models/rnn_model.h5")
rf_model = joblib.load("models/random_forest_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
meta_model = joblib.load("models/hybrid_model.pkl")

# =============================
# Load CSV and preprocess
# =============================
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df_features = df.copy()
    
    # If 'label' column exists, keep numeric labels
    if 'label' in df_features.columns:
        df_features['Actual'] = df_features['label']
        df_features = df_features.drop('label', axis=1)
    else:
        df_features['Actual'] = np.nan
    
    # Drop any unnamed index columns
    df_features = df_features.loc[:, ~df_features.columns.str.contains('^Unnamed')]
    
    return df_features

# =============================
# Predict all rows automatically
# =============================
def predict_all_rows(df_features):
    X = df_features.drop(columns=['Actual']).values
    X_scaled = scaler.transform(X)
    
    X_cnn = DLReshaper('cnn').transform(X_scaled)
    X_rnn = DLReshaper('rnn').transform(X_scaled)
    
    cnn_preds = cnn_model.predict(X_cnn)
    lstm_preds = lstm_model.predict(X_rnn)
    rnn_preds = rnn_model.predict(X_rnn)
    rf_preds = rf_model.predict_proba(X_scaled)
    svm_preds = svm_model.predict_proba(X_scaled)
    
    stacked_features = np.hstack([cnn_preds, lstm_preds, rnn_preds, rf_preds, svm_preds])
    hybrid_preds = meta_model.predict(stacked_features)
    
    # Numeric predictions only
    return hybrid_preds

# =============================
# Run script
# =============================
if __name__ == "__main__":
    csv_path = input("Enter CSV file path: ").strip()
    df_features = load_and_preprocess(csv_path)
    
    print(f"CSV loaded with {len(df_features)} samples.")
    
    predictions = predict_all_rows(df_features)
    df_features['Predicted'] = predictions  # numeric 0,1,2
    
    # Show first 20 predictions
    print("\nFirst 20 Predictions:")
    print(df_features[['Actual', 'Predicted']].head(20))
    
    # Save all predictions to CSV
    df_features.to_csv("predictions.csv", index=False)
    print("\nAll predictions saved to predictions.csv")
    
    # Interactive row inspection
    while True:
        inspect = input("\nDo you want to inspect a specific row? (y/n): ").strip().lower()
        if inspect != 'y':
            break
        try:
            row_idx = int(input(f"Enter row index (0-{len(df_features)-1}): "))
            if 0 <= row_idx < len(df_features):
                actual = df_features.at[row_idx, 'Actual']
                predicted = df_features.at[row_idx, 'Predicted']
                print(f"\nRow {row_idx} -> Actual: {actual}, Predicted: {predicted}")
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Invalid input. Enter an integer index.")
