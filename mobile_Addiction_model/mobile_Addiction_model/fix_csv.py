# mini_self_test.py
import numpy as np
import joblib
import tensorflow as tf

# Load your models and scaler
scaler = joblib.load("models/scaler.pkl")
cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
rnn_model = tf.keras.models.load_model("models/rnn_model.h5")
rf_model = joblib.load("models/random_forest_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
meta_model = joblib.load("models/hybrid_model.pkl")

class_map = {0: "Normal", 1: "Mild Addiction", 2: "Severe Addiction"}

# Custom reshaper for CNN/RNN
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

def simulate_eeg_input(choice):
    """
    Simulate a dummy EEG sample based on input choice:
    0 = Normal, 1 = Mild Addiction, 2 = Severe Addiction
    """
    if choice not in [0, 1, 2]:
        raise ValueError("Choice must be 0, 1, or 2")
    
    # Create 1 sample with 988 features
    # We'll simulate it: normal = low random, mild = mid random, severe = high random
    if choice == 0:
        sample = np.random.uniform(0, 0.3, (1, 988))
    elif choice == 1:
        sample = np.random.uniform(0.3, 0.7, (1, 988))
    else:
        sample = np.random.uniform(0.7, 1.0, (1, 988))
    
    return sample

def predict_from_array(X_new):
    # Preprocess
    X_new_scaled = scaler.transform(X_new)
    X_new_cnn = DLReshaper('cnn').transform(X_new_scaled)
    X_new_rnn = DLReshaper('rnn').transform(X_new_scaled)

    # Base model predictions
    cnn_preds = cnn_model.predict(X_new_cnn)
    lstm_preds = lstm_model.predict(X_new_rnn)
    rnn_preds = rnn_model.predict(X_new_rnn)
    rf_preds = rf_model.predict_proba(X_new_scaled)
    svm_preds = svm_model.predict_proba(X_new_scaled)

    # Stack features for meta model
    stacked_features = np.hstack([cnn_preds, lstm_preds, rnn_preds, rf_preds, svm_preds])

    # Hybrid prediction
    hybrid_preds = meta_model.predict(stacked_features)
    return [class_map[p] for p in hybrid_preds]

# ===== Interactive input =====
if __name__ == "__main__":
    print("Enter your choice to simulate EEG:")
    print("0 = Normal, 1 = Mild Addiction, 2 = Severe Addiction")
    user_choice = int(input("Your choice: "))
    
    eeg_sample = simulate_eeg_input(user_choice)
    prediction = predict_from_array(eeg_sample)
    
    print("Predicted Addiction Level:", prediction[0])
1