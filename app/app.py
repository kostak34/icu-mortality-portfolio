"""
Streamlit demo app (portfolio / educational).

What it does:
- Lets a user upload a single patient .txt file OR manually enter key observations
- Runs: parse -> feature engineering -> alignment -> imputation -> calibrated prediction
- Returns: probability + risk band

What it is NOT:
- Not a validated clinical device; demo thresholds; educational only.

Run locally (after installing requirements):
  python -m streamlit run app/app.py
"""

from __future__ import annotations

import sys
import traceback
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Robust imports for local + Streamlit Cloud
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure BOTH are importable:
# - PROJECT_ROOT enables: import scripts.step_01_load_raw
# - SCRIPTS_DIR enables older style inside your scripts: from step_01_load_raw import ...
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


# -----------------------------
# Config
# -----------------------------
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10  # demo medium threshold if not stored in bundle
MAX_UPLOAD_BYTES = 2_000_000     # 2MB safety cap for demo


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def safe_float(x: Any) -> float | None:
    """Convert to float if possible; otherwise None."""
    try:
        if x is None or x is pd.NA:
            return None
        return float(x)
    except Exception:
        return None


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def render_risk_pills(current: str) -> None:
    """Compact, not-tacky risk band display."""
    styles = {
        "LOW":    ("#14532d", "#dcfce7", "LOW"),
        "MEDIUM": ("#7c2d12", "#ffedd5", "MEDIUM"),
        "HIGH":   ("#7f1d1d", "#fee2e2", "HIGH"),
    }

    cols = st.columns(3, gap="small")
    for i, band in enumerate(["LOW", "MEDIUM", "HIGH"]):
        fg, bg, label = styles[band]
        is_active = (band == current)
        border = f"2px solid {fg}" if is_active else "1px solid #cbd5e1"
        weight = "700" if is_active else "600"
        opacity = "1.0" if is_active else "0.75"

        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    padding: 10px 10px;
                    border-radius: 999px;
                    border: {border};
                    background: {bg};
                    color: {fg};
                    text-align: center;
                    font-weight: {weight};
                    opacity: {opacity};
                    font-size: 14px;
                    ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True,
            )


def align_and_impute(
    feats: dict[str, Any],
    feature_columns: list[str],
    imputer,
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    """
    Align engineered features to training columns and impute missing values.
    Critically: uses np.nan + numeric coercion so sklearn doesn't crash on pd.NA/objects.
    """
    X = pd.DataFrame([feats])

    # Ensure expected columns exist (fill missing with np.nan, NOT pd.NA)
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep only training columns, in order
    X = X[feature_columns]

    # Convert everything to numeric; non-numeric strings -> NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    present_mask = ~X.isna().iloc[0]
    present_count = int(present_mask.sum())
    missing_features = [col for col, ok in zip(feature_columns, present_mask) if not ok]

    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)

    return X_imp, present_count, missing_features, X


def clinician_friendly_intro(thr_medium: float, thr_high: float) -> None:
    st.markdown(
        """
        **About this demo**

        This app estimates *mortality risk in ICU* from routinely collected observations and lab values.
        It’s a **portfolio demonstration** (not a validated clinical tool), designed to show how a model can be
        packaged end-to-end: data parsing → feature creation → prediction → simple risk categories.
        """
    )
    st.caption(
        f"Demo risk categories use thresholds: LOW < {thr_medium:.2f}, "
        f"MEDIUM {thr_medium:.2f}–{thr_high:.3f}, HIGH ≥ {thr_high:.3f}."
    )


def build_manual_engineered_features(
    starter_vars: list[str],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the engineered feature dict expected by the model for starter vars.

    Convention expected by your pipeline:
      - {var}_was_measured (0/1)
      - {var}_mean
      - {var}_last
    Plus some specials:
      - MechVent_prop_on, MechVent_last (optional)
      - FiO2 scaling is handled in your original pipeline; here we assume fraction if <= 1.5, % otherwise.
    """
    feats: dict[str, Any] = {}

    for var in starter_vars:
        key_last = f"{var}_last"
        key_mean = f"{var}_mean"

        last_val = inputs.get(key_last, None)
        mean_val = inputs.get(key_mean, None)

        # Normalise missing
        last_val_f = safe_float(last_val)
        mean_val_f = safe_float(mean_val)

        # If only one provided, use it for both mean & last (pragmatic for manual demo)
        if mean_val_f is None and last_val_f is not None:
            mean_val_f = last_val_f
        if last_val_f is None and mean_val_f is not None:
            last_val_f = mean_val_f

        was_measured = 1 if (last_val_f is not None or mean_val_f is not None) else 0

        feats[f"{var}_was_measured"] = was_measured
        feats[f"{var}_mean"] = mean_val_f if was_measured else np.nan
        feats[f"{var}_last"] = last_val_f if was_measured else np.nan

        # Special handling
        if var.lower() == "mechvent":
            # Clinician-friendly: Yes/No -> 1/0
            mv = inputs.get("MechVent_binary", None)
            if mv == "Yes":
                feats["MechVent_prop_on"] = 1.0
                feats["MechVent_last"] = 1.0
                feats["MechVent_mean"] = 1.0
                feats["MechVent_was_measured"] = 1
            elif mv == "No":
                feats["MechVent_prop_on"] = 0.0
                feats["MechVent_last"] = 0.0
                feats["MechVent_mean"] = 0.0
                feats["MechVent_was_measured"] = 1
            else:
                # Unknown
                feats["MechVent_prop_on"] = np.nan

        if var.lower() == "fio2":
            # Accept either fraction (0.21) or percent (21)
            # If user enters >1.5 assume percent.
            if mean_val_f is not None and mean_val_f > 1.5:
                feats[f"{var}_mean"] = mean_val_f / 100.0
            if last_val_f is not None and last_val_f > 1.5:
                feats[f"{var}_last"] = last_val_f / 100.0

    return feats


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk (Demo)", layout="centered")

st.title("ICU Mortality Risk (Demo)")
st.caption("Educational / portfolio demonstration only — **not for clinical use**.")

# Load model bundle
if not DEFAULT_BUNDLE_PATH.exists():
    st.error(
        "Model bundle not found. This app expects: `outputs/model_bundle.joblib` "
        "in the repository."
    )
    st.stop()

bundle = load_bundle(DEFAULT_BUNDLE_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
starter_vars = bundle.get("starter_vars", None)

with st.expander("About this demo", expanded=True):
    clinician_friendly_intro(thr_medium, thr_high)

# Mode selection
mode = st.radio(
    "How would you like to provide patient information?",
    options=["Upload a patient file (.txt)", "Enter values manually"],
    horizontal=True,
)

show_tech = st.checkbox("Show technical details (for debugging / portfolio)", value=False)

st.divider()


# -----------------------------
# Upload mode
# -----------------------------
if mode == "Upload a patient file (.txt)":
    uploaded = st.file_uploader("Upload a single patient `.txt` file", type=["txt"])

    if uploaded is None:
        st.info("Upload a patient file to see a prediction.")
        st.stop()

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    # Save uploaded file to temp path for your parser
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_features, X_aligned = align_and_impute(
            feats, feature_columns, imputer
        )

        prob = float(model.predict_proba(X_imp)[:, 1][0])
        band = risk_band(prob, thr_medium, thr_high)

        st.subheader("Result")
        st.metric("Predicted risk (probability)", f"{prob:.6f}")
        render_risk_pills(band)

        st.caption(
            "Risk categories are **demo thresholds** to make the output easier to interpret. "
            "They are not validated decision cut-offs."
        )

        st.subheader("Data completeness")
        st.write(f"Fields available from file: **{present_count}/{len(feature_columns)}**")
        if missing_features:
            st.write(f"Some values were missing and the model used typical values for those fields (**{len(missing_features)} imputed**).")
            with st.expander("Show missing fields"):
                st.write(missing_features)

        if show_tech:
            with st.expander("Technical details", expanded=False):
                st.write("Bundle path:", str(DEFAULT_BUNDLE_PATH))
                st.write("starter_vars stored:", "Yes" if starter_vars is not None else "No")
                st.write("First row (aligned, pre-imputation):")
                st.dataframe(X_aligned)

    except Exception as e:
        st.error(
            "We couldn’t generate a prediction from that file. "
            "This usually happens if the file is not in the expected format."
        )
        if show_tech:
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        st.stop()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -----------------------------
# Manual mode
# -----------------------------
else:
    if starter_vars is None:
        st.warning(
            "Manual entry is available, but this bundle doesn’t store `starter_vars`, "
            "so the app can’t automatically build the clinician-friendly form. "
            "Re-save the bundle with `starter_vars` included."
        )
        st.stop()

    st.subheader("Manual entry")
    st.caption(
        "Enter **most recent** values and (optionally) an **average** value. "
        "If you only enter one number, the demo uses it for both."
    )

    # Metadata for clinician-facing labels/units/help
    META = {
        "HeartRate": ("Heart rate", "bpm", "Typical adult ICU range might be ~40–160 bpm."),
        "SysBP": ("Systolic BP", "mmHg", "Enter systolic blood pressure."),
        "DiasBP": ("Diastolic BP", "mmHg", "Enter diastolic blood pressure."),
        "MeanBP": ("Mean arterial pressure", "mmHg", "Enter MAP if known."),
        "RespRate": ("Respiratory rate", "breaths/min", "Enter respiratory rate."),
        "Temp": ("Temperature", "°C", "Enter temperature in Celsius."),
        "SpO2": ("SpO₂", "%", "Oxygen saturation from pulse oximetry."),
        "SaO2": ("SaO₂ (ABG)", "%", "Arterial oxygen saturation from blood gas."),
        "FiO2": ("FiO₂", "fraction or %", "Enter 0.21 for room air or 21 for percent."),
        "pH": ("Arterial pH", "", "Typical range ~7.00–7.60."),
        "Lactate": ("Lactate", "mmol/L", "Enter lactate in mmol/L."),
        "Glucose": ("Glucose", "mmol/L", "Enter glucose in mmol/L (or leave blank)."),
        "BUN": ("BUN / Urea proxy", "", "Unit depends on dataset; leave blank if unsure."),
        "Creatinine": ("Creatinine", "", "Unit depends on dataset; leave blank if unsure."),
        "WBC": ("White cell count", "", "Unit depends on dataset; leave blank if unsure."),
        "Platelets": ("Platelets", "", "Unit depends on dataset; leave blank if unsure."),
        "HCO3": ("Bicarbonate (HCO₃⁻)", "", "Unit depends on dataset; leave blank if unsure."),
        "PaO2": ("PaO₂", "", "Unit depends on dataset; leave blank if unsure."),
        "PaCO2": ("PaCO₂", "", "Unit depends on dataset; leave blank if unsure."),
        "MechVent": ("Mechanical ventilation", "", "Select Yes/No if known."),
    }

    # Build input widgets
    inputs: dict[str, Any] = {}

    # Separate MechVent into a simple Yes/No/Unknown control
    if any(v.lower() == "mechvent" for v in starter_vars):
        mv = st.selectbox("Mechanical ventilation currently?", ["Unknown", "Yes", "No"], index=0)
        inputs["MechVent_binary"] = mv

    # Render numeric inputs for each starter var (excluding mechvent which we already handle)
    for var in starter_vars:
        if var.lower() == "mechvent":
            continue

        label, unit, help_text = META.get(var, (var, "", ""))
        st.markdown(f"**{label}** {f'({unit})' if unit else ''}")

        c1, c2 = st.columns(2, gap="small")
        with c1:
            inputs[f"{var}_last"] = st.text_input(
                f"Most recent {label}",
                value="",
                placeholder="Leave blank if unknown",
                help=help_text,
                key=f"{var}_last",
            )
        with c2:
            inputs[f"{var}_mean"] = st.text_input(
                f"Average {label} (optional)",
                value="",
                placeholder="Leave blank if unknown",
                help="Optional: if you know a typical/average value over the stay.",
                key=f"{var}_mean",
            )

        st.divider()

    # Predict button (prevents auto-reruns on every keystroke)
    if st.button("Calculate risk", type="primary"):
        # Convert manual inputs -> engineered feature dict
        raw_engineered = build_manual_engineered_features(starter_vars, inputs)

        # Friendly warnings for non-numeric entries (we coerce to NaN)
        non_numeric_fields = []
        for k, v in inputs.items():
            if k.endswith(("_last", "_mean")) and isinstance(v, str) and v.strip() != "":
                if safe_float(v) is None:
                    non_numeric_fields.append(k)

        if non_numeric_fields:
            st.warning(
                "Some entries were not recognised as numbers and were treated as missing: "
                + ", ".join(non_numeric_fields[:10])
                + (" ..." if len(non_numeric_fields) > 10 else "")
            )

        try:
            X_imp, present_count, missing_features, X_aligned = align_and_impute(
                raw_engineered, feature_columns, imputer
            )

            prob = float(model.predict_proba(X_imp)[:, 1][0])
            band = risk_band(prob, thr_medium, thr_high)

            st.subheader("Result")
            st.metric("Predicted risk (probability)", f"{prob:.6f}")
            render_risk_pills(band)

            st.caption(
                "This is a demo output. In real clinical deployment, thresholds and outputs would need "
                "formal validation, governance, and usability testing."
            )

            st.subheader("Data completeness")
            st.write(f"Values provided (after processing): **{present_count}/{len(feature_columns)}**")
            if missing_features:
                st.write(
                    f"Some values were missing and the model used typical values for those fields (**{len(missing_features)} imputed**)."
                )
                with st.expander("Show missing fields"):
                    st.write(missing_features)

            if show_tech:
                with st.expander("Technical details", expanded=False):
                    st.write("Engineered dict (manual -> model features):")
                    st.json(raw_engineered)
                    st.write("Aligned row (pre-imputation):")
                    st.dataframe(X_aligned)

        except Exception as e:
            # Clinician-friendly error, no stack trace by default
            st.error(
                "We couldn’t calculate a prediction from the values entered. "
                "Please check that fields expected to be numeric contain numbers, or leave them blank."
            )
            if show_tech:
                st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            st.stop()
