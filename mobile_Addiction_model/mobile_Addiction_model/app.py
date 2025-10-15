from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.secret_key = '1234'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ---------- Utility: Allowed file ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- Route 1: Upload EEG CSV ----------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            row_count = len(df)
            # Show only row numbers for selection
            return render_template('select_row_number.html', filename=file.filename, row_count=row_count)
        else:
            flash('Allowed file type is csv')
            return redirect(request.url)
    return render_template('upload.html')


# ---------- Route 2: Predict selected row ----------
@app.route('/predict_row', methods=['POST'])
def predict_row():
    filename = request.form.get('filename')
    row_number = int(request.form.get('row_number'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    if row_number < 0 or row_number >= len(df):
        result = 'Invalid row number selected.'
        return render_template('result.html', result=result, tips="")

    # Load models and preprocessors
    scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join('models', 'label_encoder.pkl'))
    hybrid_pipeline = joblib.load(os.path.join('models', 'hybrid_pipeline.pkl'))
    rf_model = joblib.load(os.path.join('models', 'random_forest_model.pkl'))
    svm_model = joblib.load(os.path.join('models', 'svm_model.pkl'))
    cnn_model = tf.keras.models.load_model(os.path.join('models', 'cnn_model.h5'))
    lstm_model = tf.keras.models.load_model(os.path.join('models', 'lstm_model.h5'))
    rnn_model = tf.keras.models.load_model(os.path.join('models', 'rnn_model.h5'))

    # Prepare data
    row_data = df.iloc[row_number]
    if 'label' in row_data:
        row_data = row_data.drop('label')
    if 'Label' in row_data:
        row_data = row_data.drop('Label')

    X_row = np.array(row_data).reshape(1, -1)
    X_row_scaled = scaler.transform(X_row)

    # CNN expects (samples, features, 1)
    X_row_cnn = X_row_scaled.reshape((1, X_row_scaled.shape[1], 1))
    # LSTM/RNN expects (samples, 1, features)
    X_row_rnn = X_row_scaled.reshape((1, 1, X_row_scaled.shape[1]))

    # Predictions
    cnn_pred = cnn_model.predict(X_row_cnn)
    lstm_pred = lstm_model.predict(X_row_rnn)
    rnn_pred = rnn_model.predict(X_row_rnn)
    rf_pred = rf_model.predict_proba(X_row_scaled)
    svm_pred = svm_model.predict_proba(X_row_scaled)

    # Combine all predictions
    stacked_features = np.hstack([cnn_pred, lstm_pred, rnn_pred, rf_pred, svm_pred])

    # Final hybrid prediction
    y_pred = hybrid_pipeline.predict(stacked_features)
    y_pred_int = int(np.round(y_pred)[0])

    # Label mapping
    label_map = {0: 'no addiction', 1: 'moderate', 2: 'several'}
    custom_label = label_map.get(y_pred_int, f"Unknown ({y_pred_int})")
    result = f"Prediction for row {row_number}: {custom_label}"

    # Prevention tips
    prevention_tips = {
        'no addiction': (
            "✅ Maintain your healthy habits and balanced screen usage.\n"
            "✅ Keep regular sleep schedules and avoid late-night device use.\n"
            "✅ Engage in physical activities or hobbies outside of screens.\n"
            "✅ Monitor screen time periodically to prevent overuse.\n"
            "✅ Encourage social interactions with friends and family offline."
        ),
        'moderate': (
            "⚠️ Take regular breaks from your device every 30–60 minutes.\n"
            "⚠️ Set daily screen limits and stick to them.\n"
            "⚠️ Use apps or phone settings to track and restrict usage.\n"
            "⚠️ Replace some screen time with outdoor activities or exercise.\n"
            "⚠️ Practice mindfulness or meditation to reduce compulsive use."
        ),
        'several': (
            "🚨 Seek professional help if the addiction is severe.\n"
            "🚨 Gradually reduce screen time using structured schedules.\n"
            "🚨 Engage in offline hobbies and physical exercise daily.\n"
            "🚨 Avoid triggers like notifications or excessive social media.\n"
            "🚨 Join support groups or therapy programs for digital addiction.\n"
            "🚨 Involve family or friends to monitor and motivate healthier habits."
        )
    }

    tips = prevention_tips.get(custom_label, "No tips available for this category.")
    return render_template('result.html', result=result, tips=tips)


# ---------- Route 3: Addiction Test Page ----------
@app.route('/addiction')
def test_addiction():
    return render_template('addiction.html')


# ---------- Main entry ----------
if __name__ == '__main__':
    app.run(debug=True)
