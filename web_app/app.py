from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, session
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import io
import csv
from utils.preprocess import preprocess_data
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

model = joblib.load('model/rf_model.pkl')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@app.before_request
def require_login():
    allowed_routes = ['login', 'signup', 'static', 'home']
    if request.endpoint not in allowed_routes and 'user' not in session:
        return redirect(url_for('home'))  # or login, based on design


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('signup.html', error='Username already exists')

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user'] = username
            return render_template('dashboard.html')
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        expected_features = model.feature_names_in_
        aligned_df = pd.DataFrame()
        for feature in expected_features:
            aligned_df[feature] = df[feature] if feature in df.columns else 0

        X = preprocess_data(aligned_df)
        preds = model.predict(X)
        probs = model.predict_proba(X)

        label_map = {0: "Benign", 1: "Ransomware"}
        predictions = []
        benign_count = 0
        ransom_count = 0

        for i in range(len(preds)):
            label = label_map.get(preds[i], "Unknown")
            confidence = round(max(probs[i]) * 100, 2)
            if label == "Ransomware":
                ransom_count += 1
            else:
                benign_count += 1
            predictions.append({
                'label': label,
                'class': 'ransomware' if label == "Ransomware" else 'benign',
                'probability': f"{confidence}%",
                'index': i + 1
            })

        show_graphs = False
        if 'label' in df.columns:
            y_true = df['label']
            y_pred = preds
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig('static/plots/confusion_matrix.png')
            plt.clf()

            fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend()
            plt.savefig('static/plots/roc_curve.png')
            plt.clf()

            show_graphs = True

        return render_template(
            'result.html',
            predictions=predictions,
            ransom_count=ransom_count,
            benign_count=benign_count,
            predictions_json=predictions,
            show_graphs=show_graphs
        )

    except Exception as e:
        return f"Prediction error: {e}", 500

@app.route('/download_csv', methods=['POST'])
def download_csv():
    predictions = request.json.get('predictions')
    if not predictions:
        return "No predictions available", 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['Sample Index', 'Prediction Label', 'Confidence'])
    writer.writeheader()
    for i, row in enumerate(predictions, start=1):
        writer.writerow({
            'Sample Index': i,
            'Prediction Label': row['label'],
            'Confidence': row['probability']
        })

    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)

    return send_file(mem,
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='detailed_prediction_report.csv')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
