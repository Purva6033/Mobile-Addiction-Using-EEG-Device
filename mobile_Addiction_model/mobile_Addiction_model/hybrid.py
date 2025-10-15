import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Custom Transformer for DL reshaping
# =============================
class DLReshaper(BaseEstimator, TransformerMixin):
    """
    Reshape features for CNN or RNN input.
    mode='cnn' -> (samples, features, 1)
    mode='rnn' -> (samples, 1, features)
    """
    def __init__(self, mode='cnn'):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.mode == 'cnn':
            return X.reshape((X.shape[0], X.shape[1], 1))
        elif self.mode == 'rnn':
            return X.reshape((X.shape[0], 1, X.shape[1]))
        else:
            raise ValueError("mode must be 'cnn' or 'rnn'")


# =============================
# Check label distribution
# =============================
df = pd.read_csv("balanced_dataset_EEG.csv")
print("Label distribution:")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))

# =============================
# Load Dataset
# =============================
X = df.drop(columns=['label']).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# Preprocessing
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# =============================
# Train ML Models
# =============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

svm_model = SVC(probability=True, kernel="rbf", random_state=42)
svm_model.fit(X_train_scaled, y_train)

# =============================
# Train DL Models
# =============================
X_train_cnn = DLReshaper('cnn').transform(X_train_scaled)
X_test_cnn = DLReshaper('cnn').transform(X_test_scaled)

X_train_rnn = DLReshaper('rnn').transform(X_train_scaled)
X_test_rnn = DLReshaper('rnn').transform(X_test_scaled)

num_classes = len(np.unique(y_train))

# CNN
cnn_model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
cnn_model.save("models/cnn_model.h5")

# LSTM
lstm_model = models.Sequential([
    layers.LSTM(64, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
lstm_model.save("models/lstm_model.h5")

# RNN
rnn_model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
rnn_model.save("models/rnn_model.h5")

# =============================
# Hybrid Stacking
# =============================
cnn_preds_train = cnn_model.predict(X_train_cnn)
lstm_preds_train = lstm_model.predict(X_train_rnn)
rnn_preds_train = rnn_model.predict(X_train_rnn)
rf_preds_train = rf_model.predict_proba(X_train_scaled)
svm_preds_train = svm_model.predict_proba(X_train_scaled)

cnn_preds_test = cnn_model.predict(X_test_cnn)
lstm_preds_test = lstm_model.predict(X_test_rnn)
rnn_preds_test = rnn_model.predict(X_test_rnn)
rf_preds_test = rf_model.predict_proba(X_test_scaled)
svm_preds_test = svm_model.predict_proba(X_test_scaled)

stacked_train = np.hstack([cnn_preds_train, lstm_preds_train, rnn_preds_train, rf_preds_train, svm_preds_train])
stacked_test = np.hstack([cnn_preds_test, lstm_preds_test, rnn_preds_test, rf_preds_test, svm_preds_test])

meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(stacked_train, y_train)
joblib.dump(meta_model, "models/hybrid_pipeline.pkl")

# =============================
# Evaluate
# =============================
y_pred_meta = meta_model.predict(stacked_test)
hybrid_acc = accuracy_score(y_test, y_pred_meta) * 100
print("\n✅ Hybrid Pipeline Accuracy: {:.2f}%".format(hybrid_acc))
print("\nClassification Report:\n", classification_report(y_test, y_pred_meta))

cm = confusion_matrix(y_test, y_pred_meta)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Hybrid Pipeline - Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
