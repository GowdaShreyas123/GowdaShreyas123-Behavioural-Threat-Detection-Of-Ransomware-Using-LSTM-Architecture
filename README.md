# 🔐 Real-Time Detection of Ransomware Using LSTM

This project focuses on detecting ransomware attacks in real-time using a Long Short-Term Memory (LSTM) neural network. The system monitors system-level behavior and predicts ransomware activity before significant damage occurs.

---

## 🚀 Project Overview

Ransomware is one of the most dangerous cybersecurity threats, encrypting files and demanding ransom for decryption. This project uses LSTM—a type of Recurrent Neural Network (RNN)—to analyze sequential behavioral data and detect early signs of ransomware.

---

## 🎯 Objectives

- Monitor and analyze system-level behavioral patterns
- Train an LSTM model to distinguish between normal and ransomware activity
- Predict ransomware presence in real-time
- Minimize false positives and maximize early detection

---

## 🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Jupyter Notebook**

---

## 🛠️ How It Works

1. **Data Preprocessing:** Load and clean labeled system activity logs.
2. **Model Training:** Train an LSTM model using time-series data.
3. **Prediction:** Classify sequences as benign or ransomware.
4. **Evaluation:** Measure accuracy, precision, recall, and analyze the confusion matrix.

---

## 📈 Results

- Achieved **93% accuracy** on test data
- High true positive rate with minimal false negatives
- Effective detection of unseen ransomware samples
