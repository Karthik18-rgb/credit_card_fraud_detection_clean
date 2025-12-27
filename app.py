import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, roc_auc_score

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (4, 2)
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 9

REQUIRES_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.info("This model was trained using the Kaggle Credit Card Fraud dataset. "
        "You can upload your own CSV (same structure) - or use demo mode")
st.write("This dashboard allows you to explore the performance of different machine learning models in detecting credit card fraud using the dataset and visualizing the dataset.")
@st.cache_resource
def load_model():
    return joblib.load("model/logreg_model.pkl")

model = load_model()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)
data = None

try:
    data = load_data("data/creditcard.csv")
    st.info("Loaded default dataset")
except FileNotFoundError:
    st.warning("Default dataset not found. Please upload a csv file")    

st.sidebar.header("📤 Upload CSV (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Time, Amount, V1-V28, Class)", type=["csv"])
if uploaded_file is not None:
    temp = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV preview:")
    st.dataframe(temp.head())

    missing = [c for c in REQUIRES_FEATURES if c not in temp.columns]

    if len(missing) == len(REQUIRES_FEATURES):
        st.error("❌ This CSV does not contain transaction features")
        data = data
        st.info("Switched back to default dataset.")
    else:
        for col in missing:
            temp[col] = 0  
        data = temp
        st.success("✔ Uploaded dataset loaded successfully!")

if data is None:
    try:
        data = load_data("data/sample.csv")
        st.info("🎬 Demo Mode: Loaded built-in sample dataset")    
    except FileNotFoundError:
        st.error("No dataset found. Upload a CSV to continue")
        st.stop() 

st.write("DEBUG - Data Shape: ", data.shape)
st.write("DEBUG - Columns: ", list(data.columns))

st.header("📊 Dataset Overview")
c1, c2 = st.columns(2)
with c1:
    st.write("Sample rows:")
    st.dataframe(data.head())

with c2:
    st.write("Summary")
    st.metric("Rows",len(data))
    st.metric("Features",data.shape[1])  

st.header("📈 Visualizations")

if "Class" in data.columns:
    with st.expander("📊 Fraud Class Distribution", expanded=False):
        fig, ax = plt.subplots()
        sns.countplot(x="Class", data=data, ax=ax)
        ax.set_title("Class Distribution")
        st.pyplot(fig, use_container_width=False)
        st.caption("Fraud cases are extremely rare (class imbalance problems).")

    with st.expander("💰 Transaction Amount Distribution", expanded=False):
        fig, ax = plt.subplots()
        sns.histplot(x="Amount", data=data, ax=ax, kde=False, bins=60, hue="Class")
        ax.set_title("Amount vs Fraud")
        st.pyplot(fig, use_container_width=False)
        st.caption("Large transactions sometimes show different fraud behavior.")


    with st.expander("⏱️ Fraud vs Time", expanded=False):
        sample = data.sample(min(20000, len(data)))
        fig, ax = plt.subplots(figsize=(8,4))
        sns.scatterplot(x="Time", y="Amount", hue="Class", data=sample, ax=ax, alpha=0.5, s=20)
        ax.set_title("Fraud Over Time")
        st.pyplot(fig, use_container_width=False)
        st.caption("Fraud often forms clusters at unusual times.")


    with st.expander("📉 ROC Curve & Auc", expanded=False): 
        X = data[REQUIRES_FEATURES]
        y = data["Class"]
        y_scores = model.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_scores)
        auc = roc_auc_score(y, y_scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        ax.plot([0,1], [0,1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("Receiver Operating Characteristic (ROC)")
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        st.caption("A higher AUC means the model is better at distinguishing fraud vs legitimate transactions.")
else:
    st.warning("This dataset has no 'Class' column - visualization disabled")

st.header("🕵️‍♂️ Fraud Prediction Tool")

X = data[REQUIRES_FEATURES]
index = st.slider("Select Transaction Index",0, len(X) - 1, 0, 1)
row = X.iloc[[index]]

st.write("### Transaction Details")
st.dataframe(row)

if st.button("Predict Fraud"):
    prob = model.predict_proba(row)[0][1]
    pred = model.predict(row)[0]
    st.header("🧠 Model Prediction")
    if pred == 1:
        st.error(f"⚠️ Fraud Detected - Probability: **{prob:.4f}**")
    else:
        st.success(f"✔️ Legitimate - Fraud Probability: **{prob:.4f}**")

    st.caption("Prediction made using the trained Logistic Regression pipeline")

st.markdown("---")
st.caption("Built by T Karthik Singh | Credit Card Fraud Detection ML App 🚀")        