# scripts/step_05_random_forest.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score)

import seaborn as sns
import numpy as np
from scripts.step_01_load_raw import ROOT
from scripts.step_02_batch_features import STARTER_VARS
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc, confusion_matrix, classification_report
import shap

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

# Keep the feature column order for inference later
feature_columns = list(X_train.columns)

# Fit an imputer on TRAINING data only (prevents leakage)
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test), columns=feature_columns)


# --- Build and train Random Forest ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_imp, y_train)

# --- Calibrate the model's probabilities (isotonic is best with your data volume) ---
cal = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
cal.fit(X_train_imp, y_train)

# --- Evaluate on hold-out test ---
y_pred = model.predict(X_test_imp)
y_prob = model.predict_proba(X_test_imp)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
print(f"\nAccuracy: {acc:.3f}")
print(f"ROC-AUC:  {roc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Use calibrated probabilities on the test set ---
y_prob_cal = cal.predict_proba(X_test_imp)[:, 1]

# Precision–Recall curve & area
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_cal)
pr_auc = auc(recall, precision)
print(f"\nPrecision–Recall AUC (calibrated): {pr_auc:.3f}")

# Reliability (Brier score)
brier = brier_score_loss(y_test, y_prob_cal)
print(f"Brier score (calibrated): {brier:.3f}  (lower is better)")

# Plot PR curve
plt.figure(figsize=(6, 5))
plt.step(recall, precision, where='post', label=f"PR AUC={pr_auc:.3f}")
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision (PPV)")
plt.title("Precision–Recall Curve (Calibrated RF)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Pick a threshold to target ~60% recall for deaths
target_recall = 0.60
idx = (recall >= target_recall).nonzero()[0]
if len(idx) > 0 and len(thresholds) > 0:
    chosen_idx = min(idx[-1], len(thresholds)-1)
    thr_cal = thresholds[chosen_idx]
else:
    thr_cal = 0.5  # fallback if curve never reaches target recall
print(f"\nChosen threshold for ~{target_recall*100:.0f}% recall: {thr_cal:.3f}")

# Apply threshold and show adjusted performance
y_pred_adj = (y_prob_cal >= thr_cal).astype(int)
print("\nAdjusted Confusion Matrix (calibrated @ chosen threshold):")
print(confusion_matrix(y_test, y_pred_adj))
print("\nAdjusted Classification Report (calibrated @ chosen threshold):")
print(classification_report(y_test, y_pred_adj, digits=3))

# --- Reliability / Calibration curve ---
prob_true, prob_pred = calibration_curve(y_test, y_prob_cal, n_bins=10, strategy="quantile")

plt.figure(figsize=(5,5))
plt.plot([0,1],[0,1],'--',label='Perfect')
plt.plot(prob_pred, prob_true, marker='o', label='Calibrated RF')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.show()

# --- Cross-validation for stability ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
cv_scores = cross_val_score(model, X_imp, y, cv=cv, scoring="roc_auc")
print(f"\n5-fold ROC-AUC mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# --- Feature importance plot ---
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(x=top_features, y=top_features.index, color="steelblue")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# --- SHAP Explainability (global + individual) ---

# Create SHAP explainer using the uncalibrated model (calibration changes probabilities, not feature effects)
# --- SHAP Explainability (robust across SHAP versions) ---

# --- SHAP Explainability using the modern API (works in v0.20+) ---

# Use the UNcalibrated forest for explanations
# --- SHAP Explainability using the modern API (works in v0.20+) ---

# Ensure X_test is a DataFrame with unique, string column names
# --- SHAP Explainability (robust across SHAP versions; no multi-sample Explanation) ---


# Ensure X_test is a DataFrame with clean, unique string column names
# --- SHAP Explainability (robust, no fragile SHAP plotting calls) ---

# Ensure X_test is a DataFrame with clean, unique, string column names
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
else:
    X_test = X_test.copy()
    if not X_test.columns.is_unique:
        X_test.columns = [f"{c}_{i}" for i, c in enumerate(X_test.columns)]
    X_test.columns = X_test.columns.astype(str)

# Use the UNcalibrated forest for SHAP (calibration rescales probs only)
explainer = shap.TreeExplainer(model)

# Get SHAP values in a version-robust way
sv = explainer.shap_values(X_test)
if isinstance(sv, list):          # older behavior: [class0, class1]
    sv_pos = sv[1]                # class 1 = death
else:                             # newer behavior: could be 2D or 3D
    if sv.ndim == 3:              # (n_samples, n_features, n_classes)
        sv_pos = sv[:, :, 1]      # Select class 1 (death) from last dimension
    else:                         # (n_samples, n_features)
        sv_pos = sv

# DEBUG: Check shapes
print(f"DEBUG: X_test shape: {X_test.shape}")
print(f"DEBUG: sv_pos shape after extraction: {sv_pos.shape}")

# DEBUG: Check shapes
print(f"DEBUG: X_test shape: {X_test.shape}")
print(f"DEBUG: X_test columns length: {len(X_test.columns)}")
print(f"DEBUG: sv_pos shape: {sv_pos.shape}")
print(f"DEBUG: Number of features in model: {model.n_features_in_}")

# ---- GLOBAL IMPORTANCE ----

# ---- GLOBAL IMPORTANCE: manual mean |SHAP| bar plot (top 15) ----
# ---- GLOBAL IMPORTANCE ----
mean_abs = np.abs(sv_pos).mean(axis=0).flatten()
order = np.argsort(mean_abs)[::-1][:15]
feat_names = [str(X_test.columns[i]) for i in order]
feat_values = mean_abs[order].tolist()  # Get the ordered values and convert to list

plt.figure(figsize=(8, 6))
plt.barh(feat_names[::-1], feat_values[::-1])
plt.title("Global importance: mean |SHAP| (top 15)")
plt.xlabel("Mean |SHAP| impact on death risk")
plt.tight_layout()
plt.show()

# ---- LOCAL EXPLANATIONS: straightforward bar charts for top contributors ----
# Use your chosen clinical threshold (≈0.22) on UNcalibrated probs for consistency with sv_pos
y_prob_uncal = model.predict_proba(X_test)[:, 1]
thr_uncal = 0.22
y_pred_adj = (y_prob_uncal >= thr_uncal).astype(int)
y_true = np.asarray(y_test)

tp_idx = np.where((y_true == 1) & (y_pred_adj == 1))[0]
fn_idx = np.where((y_true == 1) & (y_pred_adj == 0))[0]


def plot_local_shap_bars(i: int, title: str, k: int = 15):
    shap_vec = sv_pos[i]
    vals = np.array(shap_vec, dtype=float).flatten()  # Ensure 1D
    names = X_test.columns.tolist()

    # Make sure we don't exceed the actual number of features
    k = min(k, len(vals), len(names))

    contrib_order = np.argsort(np.abs(vals))[::-1][:k]
    vals_top = vals[contrib_order]
    names_top = [names[idx] for idx in contrib_order]

    # Get feature values - work with numpy array directly
    row_vals = X_test.iloc[i].to_numpy().flatten()  # Ensure 1D
    feature_vals_top = row_vals[contrib_order]

    # Build a DataFrame for a clean plot/print
    df_local = pd.DataFrame({
        "feature": names_top,
        "shap_value": vals_top,
        "feature_value": feature_vals_top
    }).sort_values("shap_value")

    # Plot
    plt.figure(figsize=(9, 6))
    plt.barh(df_local["feature"].tolist(), df_local["shap_value"].tolist())
    plt.title(title)
    plt.xlabel("SHAP contribution to death risk (− survival  ←  0  →  death +)")
    plt.tight_layout()
    plt.show()

    # Also print a concise table to console
    print("\nTop contributors:")
    print(df_local.sort_values("shap_value", ascending=False).to_string(index=False))

if len(tp_idx) > 0:
    plot_local_shap_bars(int(tp_idx[0]), "Local explanation — TRUE POSITIVE (death correctly flagged)")

if len(fn_idx) > 0:
    plot_local_shap_bars(int(fn_idx[0]), "Local explanation — FALSE NEGATIVE (death missed)")

# -------------------------
# SAVE MODEL BUNDLE (for demo/inference)
# -------------------------
out_dir = ROOT / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# Save the CALIBRATED model, because that's what you use for y_prob_cal and thresholding
model_to_save = cal

bundle = {
    "model": model_to_save,        # calibrated classifier
    "imputer": imputer,            # fitted on X_train
    "feature_columns": feature_columns,
    "default_threshold": float(thr_cal),  # the calibrated threshold you selected (~0.223)
    "starter_vars": STARTER_VARS,         # save feature-engineering config for reproducibility
}

joblib.dump(bundle, out_dir / "model_bundle.joblib")
print(f"\nSaved model bundle to: {out_dir / 'model_bundle.joblib'}")
print(f"Saved default threshold: {float(thr_cal):.3f}")

