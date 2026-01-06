"""
Streamlit demo app (educational).

Run:
  streamlit run app.py

Notes:
- Requires a saved model bundle at outputs/model_bundle.joblib
- Allows a user to upload a single patient file (.txt) and returns calibrated probability + risk band.
- Portfolio demo only (NOT a clinical device).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Streamlit config MUST come before any other Streamlit calls
# -------------------------------------------------------------------
st.set_page_config(page_title="ICU Mortality Demo", layout="centered")


# -------------------------------------------------------------------
# Path/import setup (robust locally + on Streamlit Cloud)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is importable so `import scripts...` works everywhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10  # your chosen medium band threshold (can be overridden by bundle)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


def align_and_impute(
    feats: dict,
    feature_columns: list[str],
    imputer,
) -> tuple[pd.DataFrame, int, list[str]]:
    """
    Create a one-row DataFrame in the exact training column order,
    report missing features, coerce numeric, and impute.
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep ONLY training columns in correct order
    X = X[feature_columns]

    # Missing reporting (pre-imputation)
    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    # Make sklearn life easier: ensure numeric-like
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Impute
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    return X_imp, present_count, missing_features


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("ICU Mortality Risk Demo (Portfolio App)")
st.caption("Educational demo only. Not for clinical use.")

with st.expander("What this is (and isn’t)"):
    st.write(
        """
        This is a **portfolio demo** using a model trained on a public ICU dataset.
        It is **not** a validated clinical device.

        The goal is to show an end-to-end ML pipeline:
        parsing → feature engineering → missingness handling → imputation → prediction → thresholds/risk bands.
        """
    )

st.caption("Uploaded files are processed for this prediction only and deleted immediately after.")

# ---- Load bundle (repo-local only; no user-provided paths) ----
if not BUNDLE_PATH.exists():
    st.error(f"Model bundle not found at: {BUNDLE_PATH.as_posix()}")
    st.stop()

bundle = load_bundle(BUNDLE_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high_default = float(bundle.get("default_threshold", 0.223))
thr_medium_default = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
starter_vars = bundle.get("starter_vars")

st.subheader("Loaded bundle")
c1, c2, c3 = st.columns(3)
c1.metric("High-risk threshold", f"{thr_high_default:.3f}")
c2.metric("Medium threshold", f"{thr_medium_default:.3f}")
c3.metric("Features expected", f"{len(feature_columns)}")

if starter_vars is None:
    st.warning("Bundle does not store `starter_vars` (feature engineering drift risk).")
else:
    st.info(f"Bundle starter_vars loaded: {len(starter_vars)} variables")

st.divider()

# ---- Sidebar: versions (nice for debugging + reproducibility) ----
with st.sidebar:
    st.write("Versions")
    st.write(f"python: {sys.version.split()[0]}")
    try:
        import sklearn  # noqa
        st.write(f"sklearn: {sklearn.__version__}")
    except Exception:
        st.write("sklearn: (not available)")
    st.write(f"numpy: {np.__version__}")
    st.write(f"pandas: {pd.__version__}")
    st.write(f"joblib: {joblib.__version__}")

    st.divider()
    st.write("Threshold controls")
    thr_high = st.slider(
        "HIGH threshold (binary)",
        min_value=0.01,
        max_value=0.99,
        value=float(thr_high_default),
        step=0.01,
    )
    thr_medium = st.slider(
        "MEDIUM threshold (risk band)",
        min_value=0.00,
        max_value=float(thr_high) - 0.01 if thr_high > 0.02 else 0.01,
        value=min(float(thr_medium_default), float(thr_high) - 0.01) if thr_high > 0.02 else 0.00,
        step=0.01,
    )

st.caption("Using bundled model: `outputs/model_bundle.joblib`")

uploaded = st.file_uploader("Upload a single patient .txt file", type=["txt"])

if uploaded is None:
    st.info("Upload a patient file to get a prediction.")
    st.stop()

# basic anti-chaos: prevent huge uploads
if uploaded.size > 2_000_000:  # 2MB
    st.error("File too large for this demo (max 2MB).")
    st.stop()

# Save upload to a true temp file (avoids collisions + avoids writing into repo folders)
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = Path(tmp.name)

st.write(f"**File received:** `{uploaded.name}`")

try:
    long_df = load_patient_long(tmp_path)
    feats = summarise_patient(long_df)

    X_imp, present_count, missing_features = align_and_impute(feats, feature_columns, imputer)

    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)
    pred = int(prob >= thr_high)

    st.subheader("Prediction")

    # Human readable
    st.metric("Predicted mortality risk", f"{prob*100:.2f}%")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Raw probability: {prob:.6f}")

    band_icon = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]
    st.write(f"**Risk band:** {band_icon} **{band} RISK**")
    st.write(f"**Binary prediction (>= HIGH threshold):** {'HIGH RISK (1)' if pred==1 else 'LOW RISK (0)'}")

    st.subheader("Feature completeness")
    st.write(f"**Features present:** {present_count}/{len(feature_columns)}")
    st.write(f"**Missing features (imputed):** {len(missing_features)}")

    if missing_features:
        with st.expander("Show missing features"):
            st.markdown("\n".join([f"- `{f}`" for f in missing_features]))

    with st.expander("Show engineered features (present only, pre-imputation)"):
        X_raw = pd.DataFrame([feats]).reindex(columns=feature_columns, fill_value=np.nan)
        present_cols = [c for c in X_raw.columns if pd.notna(X_raw.loc[0, c])]
        X_present = X_raw[present_cols].T.rename(columns={0: "value"})
        st.dataframe(X_present, use_container_width=True)

except Exception as e:
    st.error("Something went wrong while predicting.")
    st.exception(e)

finally:
    # Always delete temp file
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

