# scripts/step_04_baseline_model.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from step_01_load_raw import ROOT

# --- Load cleaned dataset ---

processed_dir = ROOT / "data" / "processed"
clean_full = processed_dir / "features_full_clean.parquet"
clean_mini = processed_dir / "mini_features_clean.parquet"

if clean_full.exists():
    df = pd.read_parquet(clean_full)
    print("Loaded CLEANED FULL dataset.")
elif clean_mini.exists():
    df = pd.read_parquet(clean_mini)
    print("Loaded CLEANED MINI dataset.")
else:
    raise FileNotFoundError("No cleaned feature table found. Run step_03_explore_and_clean.py first.")

print("Loaded cleaned dataset:", df.shape)
print("Class counts:\n", df["outcome"].value_counts())

# --- Separate features and target ---
X = df.drop(columns=["outcome", "patient_id"], errors="ignore")
y = df["outcome"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- Standardise numeric features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train baseline logistic regression ---
model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
model.fit(X_train_scaled, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
print(f"\nAccuracy: {acc:.3f}")
print(f"ROC-AUC:  {roc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Feature importance (coefficients) ---
coefs = pd.DataFrame({
    "feature": X.columns,
    "coef": model.coef_[0]
}).sort_values("coef", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(
    data=coefs.head(10), x="coef", y="feature", color="skyblue"
)
plt.title("Top Positive Predictors (↑ odds of death)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(
    data=coefs.tail(10), x="coef", y="feature", color="lightcoral"
)
plt.title("Top Negative Predictors (↓ odds of death)")
plt.tight_layout()
plt.show()
