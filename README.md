# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning web application that detects fraudulent credit-card transactions using a trained **Logistic Regression** model.

Built with **Python, scikit-learn, and Streamlit**.

---

## 🚀 Features

- Detect whether a transaction is **Fraud or Legit**
- Interactive fraud probability prediction
- Dataset insights and statistics
- Visualize fraud distribution & transaction patterns
- ROC–AUC performance visualization
- Upload custom CSV datasets (auto-validated)
- **Demo Mode** (works even without uploading data)
- Clean and user-friendly Streamlit interface

---

## 🧠 Machine Learning Model

- **Algorithm:** Logistic Regression (baseline fraud model)
- **Framework:** scikit-learn
- Handles imbalanced data using `class_weight="balanced"`
- Built using a preprocessing pipeline:
  - Median imputation
  - Standard scaling
- Trained model saved using `joblib`

### Input Features

The model uses anonymized transaction behavior features:

- `Time`
- `Amount`
- `V1 … V28` (PCA-transformed security features)

Target label:

- `Class = 0` → Legitimate  
- `Class = 1` → Fraudulent  

> The dataset is fully anonymized — no personal identities are included.

---

## 🛠️ Tech Stack

- Python  
- pandas, numpy  
- scikit-learn  
- Streamlit  
- seaborn, matplotlib  
- joblib  

---

## 📊 App Workflow

1. Load dataset (built-in demo OR uploaded CSV)  
2. Explore transaction statistics & fraud distribution  
3. Model predicts fraud probability  
4. User interacts with transactions and predictions  
5. Analyze ROC-AUC performance visually  

---

## 📥 Dataset

This project uses the public **Credit Card Fraud Detection dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- 284,807 transactions  
- 492 fraud cases (0.17%)  
- Highly imbalanced dataset  
- PCA-anonymized features (`V1–V28`)

### Using dataset locally (optional)

If you want to run with the *full dataset*:

1. Download the dataset from Kaggle  
2. Extract the ZIP  
3. Rename the CSV to:
   creditcard.csv
4. Place it inside:
   credit_card_fraud_detection_clean/
   └── data/ 
         └── creditcard.csv   

> On Streamlit Cloud, the app automatically uses a **small built-in sample dataset** (Demo Mode).

---

## 📂 Project Structure
credit_card_fraud_detection_clean/ 
│ 
├── data/ 
│   └── creditcard.csv # Full dataset (optional)
|   └── sample.csv # Demo dataset (small, included)
│
├── model/
│   └── logreg_model.pkl # Trained model 
│ 
├── train_model.py # Training script 
├── app.py # Streamlit app 
├── requirements.txt 
└── README.md

---

## ▶️ How to Run Locally
1. Install dependencies
pip install -r requirements.txt

2. (Optional) Train the model
python train_model.py

3. Run the web app
streamlit run app.py

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Real banking systems use multiple layers of security beyond ML models.

---

## 🔮 Future Enhancements

- Random Forest / XGBoost comparison  
- Precision–Recall visualization  
- Real-time anomaly detection  
- Model retraining pipeline  
- Improved UI design  

---

Developed by: **T. Karthik Singh**