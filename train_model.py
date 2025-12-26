import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

data = pd.read_csv("data/creditcard.csv")

X = data.drop("Class", axis = 1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

num_features = X.columns.tolist()

num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())])

preprocessor = ColumnTransformer(transformers=[("num", num_pipeline, num_features)])

logreg_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                  ("model", LogisticRegression(
                                      class_weight="balanced",
                                      max_iter = 1000,
                                      random_state=42
                                  ))])

rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                  ("model", RandomForestClassifier(
                                      n_estimators=100,
                                      class_weight="balanced",
                                      random_state=42,
                                      n_jobs=-1
                                  ))])


def evaluate(name, model):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return recall

print("Training Logistic Regresssion...")
logreg_pipeline.fit(X_train, y_train)
logreg_recall = evaluate("Logistic Regression", logreg_pipeline)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)
rf_recall = evaluate("Logistic Regression", rf_pipeline)

if rf_recall > logreg_recall:
    joblib.dump(rf_pipeline,"model/random_forest_model.pkl")
    print("\n Random Forest saved (higher recall)")
else:
    joblib.dump(logreg_pipeline,"model/logreg_model.pkl")
    print("\nLogistic Regression saved (higher recall)")    
