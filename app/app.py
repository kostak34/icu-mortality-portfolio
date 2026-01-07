"""
Streamlit demo app (educational portfolio).

Main module on Streamlit Cloud: app/app.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Robust imports
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10
MAX_UPLOAD_BYTES = 2_000_000  # 2MB


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def short_sha256(path: Path, n: int = 10) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def align_and_impute(
    feats: dict[str, Any],
    feature_columns: list[str],
    imputer,
) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str]]:
    """
    Returns:
      X_aligned_raw: aligned one-row df (pre-imputation)
      X_imp: imputed one-row df
      present_count: number of non-null features present
      missing_features: list of missing feature names
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep ONLY training columns in the correct order
    X = X[feature_columns].copy()

    # Missing feature reporting (pre-imputation)
    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    # Clean types for sklearn
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Impute
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

    return X, X_imp, present_count, missing_features


def predict_from_feats(
    feats: dict[str, Any],
    feature_columns: list[str],
    model,
    imputer,
    thr_medium: float,
    thr_high: float,
) -> dict[str, Any]:
    X_raw, X_imp, present_count, missing_features = align_and_impute(feats, feature_columns, imputer)

    prob = float(model.predict_proba(X_imp)[:, 1][0])
    band = risk_band(prob, thr_medium, thr_high)
    high_alert = prob >= thr_high

    return {
        "prob": prob,
        "band": band,
        "high_alert": high_alert,
        "present_count": present_count,
        "missing_features": missing_features,
        "X_raw": X_raw,
    }


def format_prob(prob: float) -> tuple[str, str]:
    # (percent_str, raw_str)
    return (f"{prob*100:.2f}%", f"{prob:.6f}")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo (Portfolio App)")
st.caption("Educational portfolio demo only — **not for clinical use**.")

with st.expander("What this is (and isn’t)"):
    st.write(
        """
This demo shows an end-to-end ML pipeline: parsing → feature engineering → imputation → prediction → risk banding.

It is **not** validated for clinical use, and results should not be used for patient care decisions.
"""
    )

# Load bundle (repo-local only)
bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(
        f"Model bundle not found at: {bundle_path}. "
        "Make sure `outputs/model_bundle.joblib` exists in the repo."
    )
    st.stop()

bundle = load_bundle(bundle_path)

model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))

starter_vars = bundle.get("starter_vars")
bundle_id = short_sha256(bundle_path)

# Simple clinician-facing headline status (no jargon)
st.subheader("Model status")
c1, c2, c3 = st.columns(3)
c1.metric("Risk bands", "LOW / MEDIUM / HIGH")
c2.metric("Features expected", f"{len(feature_columns)}")
c3.metric("Model ID", bundle_id)

# Input method
st.subheader("Input")
tabs = st.tabs(["Upload patient file (.txt)", "Manual entry (optional)"])

result: dict[str, Any] | None = None
filename_display: str | None = None
manual_mode_used = False

with tabs[0]:
    st.caption(f"Max file size for this demo: {MAX_UPLOAD_BYTES/1_000_000:.0f}MB.")
    uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("File too large for this demo (max 2MB).")
        else:
            filename_display = uploaded.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)

            try:
                long_df = load_patient_long(tmp_path)
                feats = summarise_patient(long_df)
                result = predict_from_feats(feats, feature_columns, model, imputer, thr_medium, thr_high)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

with tabs[1]:
    st.caption(
        "Enter a few values manually. Any missing features will be **imputed** by the model. "
        "This is a demo convenience mode."
    )

    # sensible defaults: a handful of common “_last” features (if they exist)
    last_feats = [c for c in feature_columns if c.endswith("_last")]
    default_pick = last_feats[:8] if len(last_feats) >= 8 else last_feats

    picked = st.multiselect(
        "Choose which engineered features to enter",
        options=feature_columns,
        default=default_pick,
        help="Tip: start with a few '_last' features. Everything else will be imputed.",
    )

    manual_inputs: dict[str, Any] = {}

    with st.form("manual_form"):
        for feat in picked:
            # allow blanks → impute later
            val = st.text_input(f"{feat}", value="", placeholder="leave blank to impute")
            if val.strip() != "":
                # best-effort numeric parse
                try:
                    manual_inputs[feat] = float(val)
                except ValueError:
                    st.warning(f"Could not parse '{feat}' as a number — it will be treated as missing.")
        submitted = st.form_submit_button("Run prediction")

    if submitted:
        manual_mode_used = True
        filename_display = "Manual entry"
        result = predict_from_feats(manual_inputs, feature_columns, model, imputer, thr_medium, thr_high)

# Show output
if result is None:
    st.info("Choose an input method above to get a prediction.")
    st.stop()

prob = float(result["prob"])
band = str(result["band"])
high_alert = bool(result["high_alert"])
present_count = int(result["present_count"])
missing_features = list(result["missing_features"])
X_raw = result["X_raw"]

pct_str, raw_str = format_prob(prob)
band_emoji = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]

st.subheader("Prediction")
st.caption(f"Source: **{filename_display}**")
st.metric("Predicted mortality risk", pct_str)
st.progress(min(max(prob, 0.0), 1.0))

st.write(f"**Risk band:** {band_emoji} **{band} RISK**")
st.write(f"**High-risk alert:** {'YES' if high_alert else 'NO'}")
st.caption(f"Raw probability: {raw_str}")

st.subheader("Data completeness")
st.write(f"**Features present:** {present_count}/{len(feature_columns)}")
st.write(f"**Missing features (imputed):** {len(missing_features)}")

if missing_features:
    with st.expander("Show missing features"):
        # nicer than a JSON blob
        st.markdown("\n".join([f"- `{m}`" for m in missing_features]))

# Engineered features: show non-null only by default
st.subheader("Engineered features")
show_all = st.checkbox("Show ALL engineered features (including missing/null)", value=False)

with st.expander("Show engineered features (aligned, pre-imputation)", expanded=True):
    aligned = X_raw.iloc[0].copy()
    if not show_all:
        aligned = aligned[aligned.notna() & (aligned.astype(str) != "None")]
    df_show = (
        aligned.reset_index()
        .rename(columns={"index": "feature", 0: "value"})
        .sort_values("feature")
        .reset_index(drop=True)
    )
    st.dataframe(df_show, use_container_width=True)

# Technical details (hidden by default)
with st.expander("Technical details (for portfolio / debugging)"):
    st.write(f"- Bundle path (repo): `outputs/model_bundle.joblib`")
    st.write(f"- Bundle SHA256 (short): `{bundle_id}`")
    st.write(f"- Thresholds: LOW < {thr_medium:.3f}, MEDIUM [{thr_medium:.3f}..{thr_high:.3f}), HIGH ≥ {thr_high:.3f}")
    if starter_vars is None:
        st.write("- `starter_vars`: not stored in bundle")
    else:
        st.write(f"- `starter_vars`: {len(starter_vars)} variables")

    st.write(f"- Python: {sys.version.split()[0]}")
    try:
        import sklearn  # type: ignore
        st.write(f"- scikit-learn: {sklearn.__version__}")
    except Exception:
        pass
    st.write(f"- numpy: {np.__version__}")
    st.write(f"- pandas: {pd.__version__}")
    st.write(f"- joblib: {joblib.__version__}")

if manual_mode_used:
    st.warning(
        "Manual entry mode used: some/many features were imputed. "
        "This is expected in this demo mode."
    )
