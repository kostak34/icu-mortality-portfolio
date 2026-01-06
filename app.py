"""
Streamlit demo app (educational).

Run:
  streamlit run app.py

Notes:
- Requires a saved model bundle at outputs/model_bundle.joblib
- Allows a user to upload a single patient file (.txt) and returns risk probability + label.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore

st.set_page_config(page_title="ICU Mortality Demo", layout="centered")

st.title("ICU Mortality Prediction (Demo)")
st.caption("Educational demo only. Not for clinical use.")

bundle_path = Path("outputs/model_bundle.joblib")
if not bundle_path.exists():
    st.error("Model bundle not found at outputs/model_bundle.joblib. Train and save the bundle first.")
    st.stop()

bundle = joblib.load(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]
default_thr = float(bundle.get("default_threshold", 0.223))

thr = st.slider("Decision threshold", min_value=0.01, max_value=0.99, value=float(default_thr), step=0.01)

uploaded = st.file_uploader("Upload a single patient .txt file", type=["txt"])
if uploaded is None:
    st.info("Upload a patient file to get a prediction.")
    st.stop()

tmp_path = Path("outputs/_tmp_uploaded_patient.txt")
tmp_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path.write_bytes(uploaded.getvalue())

long_df = load_patient_long(tmp_path)
feats = summarise_patient(long_df)

X = pd.DataFrame([feats])
for c in feature_columns:
    if c not in X.columns:
        X[c] = pd.NA
X = X[feature_columns]
X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

prob = float(model.predict_proba(X_imp)[:, 1][0])
pred = int(prob >= thr)

st.subheader("Result")
st.metric("Predicted mortality risk (probability)", f"{prob:.3f}")
st.write("Prediction:", "HIGH RISK (1)" if pred==1 else "LOW RISK (0)")

with st.expander("Show engineered features"):
    st.dataframe(X_imp)
