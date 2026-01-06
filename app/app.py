# app/app.py
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import sklearn
import numpy as numpy_pkg  # just to show version cleanly

# -----------------------------
# Path setup (robust imports)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is importable so `import scripts...` works everywhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long
from scripts.step_02_batch_features import summarise_patient


DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10
MAX_UPLOAD_BYTES = 2_000_000  # 2MB


@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def align_and_impute(
    feats: dict, feature_columns: list[str], imputer
) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str]]:
    """
    Returns:
      X_imp: imputed 1-row DF in training column order
      X_aligned_raw: aligned 1-row DF (pre-imputation)
      present_count: number of non-missing features (pre-imputation)
      missing_features: list[str] of missing feature names (pre-imputation)
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep ONLY training columns in the correct order
    X = X[feature_columns]

    # Missing feature reporting (pre-imputation)
    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    # Coerce to numeric safely (anything weird becomes NaN)
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    X_aligned_raw = X.copy()

    # Impute
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

    return X_imp, X_aligned_raw, present_count, missing_features


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

# Sidebar: versions + controls
with st.sidebar:
    st.header("App info")
    st.caption("Educational portfolio demo. Not for clinical use.")

    st.subheader("Environment")
    st.write(f"Python: {sys.version.split()[0]}")
    st.write(f"scikit-learn: {sklearn.__version__}")
    st.write(f"numpy: {numpy_pkg.__version__}")
    st.write(f"pandas: {pd.__version__}")
    st.write(f"joblib: {joblib.__version__}")

    st.divider()
    st.subheader("Display")
    show_all_features = st.checkbox("Show ALL engineered features", value=False)
    show_missing_as_table = st.checkbox("Show missing features as a table", value=False)


st.title("ICU Mortality Risk Demo (Portfolio App)")
st.caption("Upload a single patient `.txt` file and get a calibrated probability + risk band.")

with st.expander("What this is (and isn’t)", expanded=False):
    st.write(
        """
        This is a **portfolio demo** using a model trained on a public ICU dataset.
        It is **not** a validated clinical device. It’s meant to show end-to-end ML delivery:
        parsing → features → imputation → prediction → thresholding → risk bands.
        """
    )

# Load the bundled model from the repo only
bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(
        "Model bundle not found in the expected location: "
        f"{bundle_path}. Make sure `outputs/model_bundle.joblib` is committed."
    )
    st.stop()

bundle = load_bundle(bundle_path)

# Pull items from bundle
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
starter_vars = bundle.get("starter_vars", None)

# Nice-looking path (avoid /mount/src…)
try:
    nice_bundle_path = str(bundle_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
except Exception:
    nice_bundle_path = "outputs/model_bundle.joblib"

st.subheader("Loaded bundle")
c1, c2, c3 = st.columns(3)
c1.metric("High-risk threshold", f"{thr_high:.3f}")
c2.metric("Medium threshold", f"{thr_medium:.3f}")
c3.metric("Features expected", f"{len(feature_columns)}")

st.caption(f"Using bundled model: `{nice_bundle_path}`")

if starter_vars is None:
    st.warning("Bundle does not store `starter_vars` (feature engineering drift risk).")
else:
    st.info(f"Bundle starter_vars loaded: {len(starter_vars)} variables")

st.divider()

st.subheader("Upload")
st.caption("Max file size for this demo: **2MB** (even if Streamlit shows a higher default limit).")
uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])

if uploaded is None:
    st.stop()

# Enforce your real limit
if uploaded.size > MAX_UPLOAD_BYTES:
    st.error("File too large for this demo (max 2MB).")
    st.stop()

st.write(f"**File received:** `{uploaded.name}`")

# Save uploaded file to a temp location because your loader expects a path
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = Path(tmp.name)

try:
    long_df = load_patient_long(tmp_path)
    feats = summarise_patient(long_df)

    X_imp, X_aligned_raw, present_count, missing_features = align_and_impute(
        feats, feature_columns, imputer
    )

    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)
    binary_pred = int(prob >= thr_high)

    # --- Prediction block ---
    st.subheader("Prediction")
    st.metric("Predicted mortality risk (probability)", f"{prob:.6f}")
    st.progress(clamp01(prob))

    band_icon = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]
    st.write(f"**Risk band:** {band_icon} **{band} RISK**")
    st.write(
        f"**Binary prediction (>= HIGH threshold):** "
        f"{'HIGH RISK (1)' if binary_pred==1 else 'LOW RISK (0)'}"
    )

    # --- Feature completeness ---
    st.subheader("Feature completeness")
    st.write(f"**Features present:** {present_count}/{len(feature_columns)}")
    st.write(f"**Missing features (imputed):** {len(missing_features)}")

    if missing_features:
        with st.expander("Show missing features", expanded=False):
            if show_missing_as_table:
                st.dataframe(
                    pd.DataFrame({"missing_feature": missing_features}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                # cleaner than the indexed JSON-looking list
                st.markdown("\n".join([f"- `{m}`" for m in missing_features]))

    # --- Engineered features display ---
    st.subheader("Engineered features")
    with st.expander("Show engineered features (aligned, pre-imputation)", expanded=False):
        if (starter_vars is not None) and (not show_all_features):
            # Show only starter vars + their derived fields (mean/last/was_measured etc.)
            starter_prefixes = set([v.lower() for v in starter_vars])
            cols_to_show = [
                c for c in X_aligned_raw.columns
                if c.split("_")[0].lower() in starter_prefixes
            ]
            view = X_aligned_raw[cols_to_show] if cols_to_show else X_aligned_raw
            st.caption("Showing starter-variable-derived features (toggle sidebar to show all).")
        else:
            view = X_aligned_raw

        # Tall table = readable on web
        long_view = (
            view.iloc[0]
            .rename("value")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        st.dataframe(long_view, use_container_width=True, hide_index=True)

except Exception as e:
    st.error("Something went wrong while predicting.")
    st.exception(e)
finally:
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
