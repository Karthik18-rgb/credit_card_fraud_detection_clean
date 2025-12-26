# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning web application that detects fraudulent credit-card transactions using a trained Logistic Regression model.
Built with Python, scikit-learn, and Streamlit.

## 🚀 Features

- Detect whether a transaction is Fraud or Legit
- View dataset insights and statistics
- Visualize fraud distribution & patterns
- Interactive ROC-AUC performance visualization
- Upload custom CSV datasets (auto-validated)
- Clean and user-friendly Streamlit interface

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression (baseline fraud model)
- Framework: scikit-learn
- Handles highly imbalanced data using class_weight="balanced"
- Uses preprocessing pipeline with: 
- Median imputation
- Standard scaling
- Model saved and loaded using joblib

### Input Features
- (Automatically generated PCA transaction features)
- Time
- Amount
- V1 … V28
- Target Label
- Class = 0 → Legit
- Class = 1 → Fraud

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn
- Streamlit
- seaborn, matplotlib
- joblib

## 📊 App Workflow

1. Load dataset (or upload CSV)
2. Visualize fraud distribution & transaction behavior
3. ML model predicts fraud probability
4. User explores predictions interactively
5. Optional: analyze ROC curve & AUC score

## ⚠️ Disclaimer

This system is intended for educational purposes only.
Real banking systems combine multiple layers of security, rules, and monitoring— this demo focuses on the ML component only.

📥 Dataset
This project uses the public Credit Card Fraud Detection dataset from Kaggle.
Download it here:
https://www.kaggle.com/mlg-ulb/creditcardfraud
After downloading:
Extract the ZIP file
Rename the CSV (if needed) to:
creditcard.csv
Place it inside the project folder:
Copy code

credit_card_fraud_detection_clean/
 └── data/
     └── creditcard.csv

## 📂 Project Structure

credit_card_fraud_detection_clean/ 
├── data/ 
│ └── creditcard.csv 
│
├── model/ 
│ └── model.pkl 
│
├── train_model.py
├── app.py 
├── requirements.txt 
└── README.md 

## ▶ How to Run

1. Install dependencies:
   pip install -r requirements.txt 
2. Train the model (optional — model is included):
   python train_model.py 
3. Run the app:
   streamlit run app.py 
