# app/app.py
"""
ICU Mortality Risk Demo (Portfolio)

Clinician-facing demo UI around a pre-trained model bundle.
- Upload a single patient .txt (same format as your dataset)
- OR enter observations manually at 12h / 6h / Now

Important:
- Demo / education only. Not for clinical use.
- In "Estimate mode", missing model inputs are filled using training-set typical values
  (via the saved imputer). This is clearly disclosed to the user.
- In "Strict mode" (default), the app will refuse to predict if there isn't enough data.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Robust imports: make project root importable so `import scripts...` works on
# Streamlit Cloud (main module is app/app.py).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"
DEFAULT_MEDIUM_THRESHOLD = 0.10
STRICT_MIN_PRESENT_FEATURES = 20  # guardrail; tune if needed
STRICT_MIN_MEASUREMENTS_ENTERED = 4  # guardrail; tune if needed
MAX_UPLOAD_BYTES = 2_000_000  # 2MB


# -----------------------------------------------------------------------------
# Clinician-friendly labels + ranges for manual entry
# -----------------------------------------------------------------------------
def _label(base: str) -> str:
    return {
        "HeartRate": "Heart rate (bpm)",
        "MeanBP": "Mean arterial pressure (NIBP) (mmHg)",
        "SysBP": "Systolic BP (NIBP) (mmHg)",
        "DiasBP": "Diastolic BP (NIBP) (mmHg)",
        "RespRate": "Respiratory rate (breaths/min)",
        "SaO2": "SaO₂ (%)",
        "FiO2": "FiO₂ (fraction) — enter 0.21 to 1.00",
        "MechVent": "Mechanical ventilation",
        "pH": "pH",
        "HCO3": "HCO₃⁻ (mmol/L)",
        "Lactate": "Lactate (mmol/L)",
        "Glucose": "Glucose (mmol/L)",
        "Na": "Sodium (Na) (mmol/L)",
        "K": "Potassium (K) (mmol/L)",
        "Mg": "Magnesium (Mg) (mmol/L)",
        "Creatinine": "Creatinine (µmol/L)",
        "BUN": "Urea (BUN) (mg/dL)",
        "Platelets": "Platelets (×10⁹/L)",
        "HCT": "Haematocrit (%)",
        "GCS": "GCS",
    }.get(base, base)


RANGES: Dict[str, Tuple[float, float, str]] = {
    "HeartRate": (0, 250, "bpm"),
    "MeanBP": (0, 200, "mmHg"),
    "SysBP": (0, 300, "mmHg"),
    "DiasBP": (0, 200, "mmHg"),
    "RespRate": (0, 80, "breaths/min"),
    "SaO2": (0, 100, "%"),
    "FiO2": (0.21, 1.0, "fraction"),
    "pH": (6.8, 7.8, ""),
    "HCO3": (0, 60, "mmol/L"),
    "Lactate": (0, 30, "mmol/L"),
    "Glucose": (0, 60, "mmol/L"),
    "Na": (90, 200, "mmol/L"),
    "K": (0, 10, "mmol/L"),
    "Mg": (0, 5, "mmol/L"),
    "Creatinine": (0, 2000, "µmol/L"),
    "BUN": (0, 200, "mg/dL"),
    "Platelets": (0, 2000, "×10⁹/L"),
    "HCT": (0, 80, "%"),
    "GCS": (3, 15, ""),
}


# -----------------------------------------------------------------------------
# Bundle loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_bundle(path: Path) -> Dict[str, Any]:
    return joblib.load(path)


# -----------------------------------------------------------------------------
# Helpers: missing features formatting (clinician-friendly)
# -----------------------------------------------------------------------------
def _format_feature_name(model_feature: str) -> str:
    # Turn "RespRate_mean" -> "Respiratory rate (breaths/min) — average"
    suffix_map = {"_mean": "average", "_last": "latest", "_was_measured": "recorded"}
    for suf, human in suffix_map.items():
        if model_feature.endswith(suf):
            base = model_feature[: -len(suf)]
            return f"{_label(base)} — {human}"

    if model_feature == "MechVent_prop_on":
        return f"{_label('MechVent')} — proportion on"

    # fallback: de-snake
    return model_feature.replace("_", " ")


def format_missing_features(missing_features: List[str]) -> List[str]:
    pretty = [_format_feature_name(f) for f in missing_features]
    # unique but stable
    seen = set()
    out = []
    for p in pretty:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


# -----------------------------------------------------------------------------
# Core: align + impute safely (NO pd.NA leakage into sklearn)
# -----------------------------------------------------------------------------
def align_and_impute(
    feats: Dict[str, Any],
    feature_columns: List[str],
    imputer,
) -> Tuple[pd.DataFrame, int, List[str], pd.DataFrame]:
    """
    Returns:
      X_imp: imputed DataFrame (model input)
      present_count: non-missing model features BEFORE imputation
      missing_features: list[str] of model feature names missing pre-imputation
      X_aligned: aligned numeric DataFrame pre-imputation (for transparency)
    """
    X = pd.DataFrame([feats])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_columns]

    # Normalize missing markers (IMPORTANT)
    X = X.replace({pd.NA: np.nan})

    # Force numeric; anything else becomes NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    X_aligned = X.copy()

    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    return X_imp, present_count, missing_features, X_aligned


# -----------------------------------------------------------------------------
# Explainability: always return something useful (SHAP if possible, else fallback)
# -----------------------------------------------------------------------------
def _get_explain_model(bundle: Dict[str, Any]):
    m = bundle["model"]
    # If it's calibrated, try to pull underlying estimator used in calibrator
    if hasattr(m, "calibrated_classifiers_") and getattr(m, "calibrated_classifiers_", None):
        try:
            return m.calibrated_classifiers_[0].estimator
        except Exception:
            return m
    return m


def compute_top_drivers(
    bundle: Dict[str, Any],
    X_imp: pd.DataFrame,
    X_aligned: pd.DataFrame,
    feature_columns: List[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Returns a table of top factors that *push risk up* for this patient.
    If SHAP fails, falls back to global feature_importances_.
    """
    explain_model = _get_explain_model(bundle)

    # Try SHAP
    try:
        import shap  # lazy import
        explainer = shap.TreeExplainer(explain_model)
        sv = explainer.shap_values(X_imp)

        if isinstance(sv, list):
            sv_pos = np.asarray(sv[1])[0]  # class 1
        else:
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:
                sv_pos = sv_arr[0, :, 1]
            else:
                sv_pos = sv_arr[0]

        rows = []
        for i, f in enumerate(feature_columns):
            shap_val = float(sv_pos[i])
            if shap_val <= 0:
                continue  # ONLY pushes risk up

            raw_val = X_aligned.iloc[0][f]
            imp_val = float(X_imp.iloc[0][i])
            is_imputed = pd.isna(raw_val)
            shown_val = imp_val if is_imputed else float(raw_val)

            rows.append(
                {
                    "Factor": _format_feature_name(f),
                    "Value used": shown_val,
                    "Imputed?": "Yes" if is_imputed else "No",
                    "Pushes risk by": shap_val,
                }
            )

        df = pd.DataFrame(rows).sort_values("Pushes risk by", ascending=False).head(top_n)
        if not df.empty:
            return df.reset_index(drop=True)

        # If nothing positive, show top absolute influence (still patient-specific)
        rows = []
        for i, f in enumerate(feature_columns):
            shap_val = float(sv_pos[i])
            raw_val = X_aligned.iloc[0][f]
            imp_val = float(X_imp.iloc[0][i])
            is_imputed = pd.isna(raw_val)
            shown_val = imp_val if is_imputed else float(raw_val)
            rows.append(
                {
                    "Factor": _format_feature_name(f),
                    "Value used": shown_val,
                    "Imputed?": "Yes" if is_imputed else "No",
                    "Influence (abs)": abs(shap_val),
                }
            )
        df2 = pd.DataFrame(rows).sort_values("Influence (abs)", ascending=False).head(top_n)
        return df2.reset_index(drop=True)

    except Exception:
        # Fallback: global feature importance
        if hasattr(explain_model, "feature_importances_"):
            imp = np.asarray(explain_model.feature_importances_, dtype=float)
            order = np.argsort(imp)[::-1][:top_n]
            rows = []
            for idx in order:
                f = feature_columns[idx]
                raw_val = X_aligned.iloc[0][f]
                imp_val = float(X_imp.iloc[0][idx])
                is_imputed = pd.isna(raw_val)
                shown_val = imp_val if is_imputed else float(raw_val)
                rows.append(
                    {
                        "Factor": _format_feature_name(f),
                        "Value used": shown_val,
                        "Imputed?": "Yes" if is_imputed else "No",
                        "Model importance": float(imp[idx]),
                    }
                )
            return pd.DataFrame(rows)

        return pd.DataFrame([{"Factor": "Explanation unavailable."}])


# -----------------------------------------------------------------------------
# Risk banding
# -----------------------------------------------------------------------------
def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def band_badge(band: str) -> str:
    return {"LOW": "🟢 LOW RISK", "MEDIUM": "🟠 MEDIUM RISK", "HIGH": "🔴 HIGH RISK"}[band]


# -----------------------------------------------------------------------------
# Manual entry parsing + engineering into model features
# -----------------------------------------------------------------------------
def _parse_numeric(s: str, base: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a numeric string, apply range checks, return (value, warning_message).
    Blank => (None, None)
    Non-numeric => (None, warning)
    Out of range => (None, warning)
    """
    s = (s or "").strip()
    if s == "":
        return None, None

    try:
        v = float(s)
    except Exception:
        lo, hi, unit = RANGES.get(base, (None, None, ""))
        unit_txt = f" ({unit})" if unit else ""
        return None, f"Please enter a number for {_label(base)}{unit_txt}."

    if base in RANGES:
        lo, hi, unit = RANGES[base]
        if v < lo or v > hi:
            unit_txt = f" {unit}".strip()
            unit_txt = f" {unit_txt}" if unit_txt else ""
            return None, f"Please enter {_label(base)} between {lo:g} and {hi:g}{unit_txt}."
    return v, None


def _parse_yesno(s: str) -> Optional[int]:
    if s == "Yes":
        return 1
    if s == "No":
        return 0
    return None


def engineer_from_manual(
    manual: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, List[str], Dict[str, List[str]]]:
    """
    manual[base] = {"12h": val/None, "6h": val/None, "now": val/None}
    Returns:
      feats dict for model alignment
      measurements_entered count (count of base variables with ANY input)
      warnings list (clinician-friendly)
      missing_by_time dict for UI (what user didn't enter)
    """
    feats: Dict[str, Any] = {}
    warnings: List[str] = []
    missing_by_time: Dict[str, List[str]] = {"12h": [], "6h": [], "now": []}

    measurements_entered = 0

    for base, tvals in manual.items():
        v12 = tvals.get("12h", None)
        v6 = tvals.get("6h", None)
        vnow = tvals.get("now", None)

        # Track missing by time (for UI only)
        if v12 is None:
            missing_by_time["12h"].append(_label(base))
        if v6 is None:
            missing_by_time["6h"].append(_label(base))
        if vnow is None:
            missing_by_time["now"].append(_label(base))

        vals = [v for v in [v12, v6, vnow] if v is not None]

        if len(vals) == 0:
            feats[f"{base}_was_measured"] = 0
            feats[f"{base}_mean"] = np.nan
            feats[f"{base}_last"] = np.nan
            continue

        measurements_entered += 1
        feats[f"{base}_was_measured"] = 1

        mean_val = float(np.mean(vals))
        last_val = vnow if vnow is not None else (v6 if v6 is not None else v12)

        feats[f"{base}_mean"] = mean_val
        feats[f"{base}_last"] = float(last_val) if last_val is not None else np.nan

        # special handling
        if base == "MechVent":
            mv = np.array(vals, dtype=float)
            feats["MechVent_prop_on"] = float(np.mean(mv)) if mv.size else np.nan
            feats["MechVent_last"] = float(last_val) if last_val is not None else np.nan

    return feats, measurements_entered, warnings, missing_by_time


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo")
st.caption("For demonstration and education only — not a clinical decision tool.")

# Bundle must exist in repo
if not BUNDLE_PATH.exists():
    st.error("Model bundle not found. Expected: outputs/model_bundle.joblib")
    st.stop()

bundle = load_bundle(BUNDLE_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns: List[str] = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
thr_low = float(bundle.get("low_threshold", thr_medium))  # LOW is < thr_medium

# Keep input method stable across reruns
if "input_method" not in st.session_state:
    st.session_state["input_method"] = "Manual entry"

st.markdown("### Choose input method")
method = st.radio(
    label="",
    options=["Upload patient file", "Manual entry"],
    index=0 if st.session_state["input_method"] == "Upload patient file" else 1,
    key="input_method",
    horizontal=True,
)

st.divider()

with st.expander("What this tool does", expanded=False):
    st.write(
        "It estimates mortality risk using a model trained on a public ICU dataset. "
        "It can help you understand how an ML workflow turns observations into a risk estimate."
    )
    st.write(
        "It **does not** replace clinical judgement. The thresholds are demonstration cut-offs and not validated for practice."
    )

with st.expander("What data can I upload?", expanded=False):
    st.write(
        "Upload a single patient `.txt` file from the same public dataset format used to train this demo. "
        "For manual entry, enter values at 12h / 6h / Now. Leave unknown fields blank."
    )

# Prediction mode (DEFAULT STRICT)
st.markdown("### Prediction mode")
mode = st.radio(
    "",
    ["Strict (recommended)", "Estimate (fills missing with typical training values)"],
    index=0,
    horizontal=False,
)
strict_mode = mode.startswith("Strict")

if not strict_mode:
    st.info(
        "Estimate mode is ON: missing model inputs will be filled using typical values learned during training. "
        "This will be clearly marked in the results."
    )

# -----------------------------------------------------------------------------
# UPLOAD MODE
# -----------------------------------------------------------------------------
if method == "Upload patient file":
    uploaded = st.file_uploader("Upload patient file (.txt)", type=["txt"])

    if uploaded is None:
        st.stop()

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_features, X_aligned = align_and_impute(feats, feature_columns, imputer)

        # Strict mode guardrail
        if strict_mode:
            if present_count < STRICT_MIN_PRESENT_FEATURES:
                st.error(
                    f"Insufficient data to make a reliable estimate in Strict mode.\n\n"
                    f"- Model features available: {present_count}/{len(feature_columns)}\n"
                    f"- Minimum required (demo rule): {STRICT_MIN_PRESENT_FEATURES}\n\n"
                    "Add more data (or switch to Estimate mode)."
                )
                st.stop()

        prob = float(model.predict_proba(X_imp)[:, 1][0])
        band = risk_band(prob, thr_medium, thr_high)

        st.subheader("Result")
        st.write(band_badge(band))
        st.metric("Estimated mortality risk", f"{prob*100:.2f}%")
        st.caption(f"Model probability: {prob:.6f}")
        st.caption(f"Risk bands: LOW < {thr_medium:.2f}, MEDIUM {thr_medium:.2f}–{thr_high:.3f}, HIGH ≥ {thr_high:.3f}")

        # Missing (specific, clinician-friendly)
        st.subheader("What data was missing (and filled in by the model)")
        missing_pretty = format_missing_features(missing_features)
        if len(missing_pretty) == 0:
            st.write("None — all model features were available from the uploaded data.")
        else:
            if strict_mode:
                st.write(
                    "Some model inputs were missing. In Strict mode, the app only proceeds if enough data is present overall."
                )
            else:
                st.write("Missing inputs were filled using typical training values (Estimate mode).")

            for item in missing_pretty:
                st.write(f"• {item}")

        # Explainability (always table)
        st.subheader("Why this risk estimate?")
        st.caption("Top factors that increased the model’s estimate. This is not proof of causation.")
        drivers_df = compute_top_drivers(bundle, X_imp, X_aligned, feature_columns, top_n=5)
        st.dataframe(drivers_df, use_container_width=True, hide_index=True)

        # Transparency: engineered values available (no nested expanders!)
        with st.expander("Engineered values (one row, pre-imputation)", expanded=False):
            # show only non-null engineered values
            row = pd.DataFrame([feats])
            row = row.replace({pd.NA: np.nan})
            nonnull = row.loc[:, row.notna().any(axis=0)]
            st.dataframe(nonnull, use_container_width=True)

        with st.expander("Technical details", expanded=False):
            st.write(f"Bundle: {BUNDLE_PATH.as_posix()}")
            st.write(f"Features expected: {len(feature_columns)}")
            st.write(f"High threshold: {thr_high:.3f}")
            st.write(f"Medium threshold: {thr_medium:.2f}")

    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.exception(e)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# MANUAL MODE
# -----------------------------------------------------------------------------
if method == "Manual entry":
    st.subheader("Manual entry")
    st.caption("Enter observations at up to three timepoints. Leave blank if unknown.")

    # Manual entry UI helpers
    def _row_numeric(base: str, key_prefix: str):
        c0, c1, c2, c3 = st.columns([2.2, 1.2, 1.2, 1.2])
        c0.write(_label(base))

        v12 = c1.text_input("", key=f"{key_prefix}_{base}_12h", placeholder="", label_visibility="collapsed")
        v6 = c2.text_input("", key=f"{key_prefix}_{base}_6h", placeholder="", label_visibility="collapsed")
        vnow = c3.text_input("", key=f"{key_prefix}_{base}_now", placeholder="", label_visibility="collapsed")

        parsed = {}
        for t, raw in [("12h", v12), ("6h", v6), ("now", vnow)]:
            val, warn = _parse_numeric(raw, base)
            if warn:
                st.warning(warn)
            parsed[t] = val
        return parsed

    def _row_yesno(base: str, key_prefix: str):
        c0, c1, c2, c3 = st.columns([2.2, 1.2, 1.2, 1.2])
        c0.write(_label(base))
        v12 = c1.selectbox("", ["Unknown", "No", "Yes"], key=f"{key_prefix}_{base}_12h", label_visibility="collapsed")
        v6 = c2.selectbox("", ["Unknown", "No", "Yes"], key=f"{key_prefix}_{base}_6h", label_visibility="collapsed")
        vnow = c3.selectbox("", ["Unknown", "No", "Yes"], key=f"{key_prefix}_{base}_now", label_visibility="collapsed")
        return {"12h": _parse_yesno(v12), "6h": _parse_yesno(v6), "now": _parse_yesno(vnow)}

    st.write("")  # spacing

    st.markdown("#### Cardiac")
    st.caption("Measurement | 12h ago | 6h ago | Now")
    manual_inputs: Dict[str, Dict[str, Any]] = {}
    manual_inputs["HeartRate"] = _row_numeric("HeartRate", "m")
    manual_inputs["MeanBP"] = _row_numeric("MeanBP", "m")
    manual_inputs["SysBP"] = _row_numeric("SysBP", "m")
    manual_inputs["DiasBP"] = _row_numeric("DiasBP", "m")

    st.markdown("#### Respiratory")
    st.caption("Measurement | 12h ago | 6h ago | Now")
    manual_inputs["RespRate"] = _row_numeric("RespRate", "m")
    manual_inputs["SaO2"] = _row_numeric("SaO2", "m")
    manual_inputs["FiO2"] = _row_numeric("FiO2", "m")
    manual_inputs["MechVent"] = _row_yesno("MechVent", "m")

    st.markdown("#### Blood gas")
    st.caption("Measurement | 12h ago | 6h ago | Now")
    manual_inputs["pH"] = _row_numeric("pH", "m")
    manual_inputs["HCO3"] = _row_numeric("HCO3", "m")

    st.markdown("#### Bloods")
    st.caption("Measurement | 12h ago | 6h ago | Now")
    manual_inputs["Na"] = _row_numeric("Na", "m")
    manual_inputs["K"] = _row_numeric("K", "m")
    manual_inputs["Mg"] = _row_numeric("Mg", "m")
    manual_inputs["Creatinine"] = _row_numeric("Creatinine", "m")
    manual_inputs["BUN"] = _row_numeric("BUN", "m")
    manual_inputs["Platelets"] = _row_numeric("Platelets", "m")
    manual_inputs["HCT"] = _row_numeric("HCT", "m")
    manual_inputs["Glucose"] = _row_numeric("Glucose", "m")
    manual_inputs["Lactate"] = _row_numeric("Lactate", "m")

    st.markdown("#### Other")
    st.caption("Measurement | 12h ago | 6h ago | Now")
    manual_inputs["GCS"] = _row_numeric("GCS", "m")

    st.divider()

    # Calculate button
    if st.button("Calculate risk", type="primary"):
        try:
            feats, measurements_entered, warnings, missing_by_time = engineer_from_manual(manual_inputs)

            X_imp, present_count, missing_features, X_aligned = align_and_impute(feats, feature_columns, imputer)

            st.subheader("Data check")
            st.write(f"Measurements entered: **{measurements_entered}**")
            st.write(f"Model features available (derived from those): **{present_count}/{len(feature_columns)}**")
            missing_count = len(missing_features)
            st.write(f"Missing model features (filled if Estimate mode): **{missing_count}**")

            # Strict mode guardrail
            if strict_mode:
                if present_count < STRICT_MIN_PRESENT_FEATURES or measurements_entered < STRICT_MIN_MEASUREMENTS_ENTERED:
                    st.error(
                        "Insufficient data to make a reliable estimate in Strict mode.\n\n"
                        f"- Measurements entered: {measurements_entered} (min {STRICT_MIN_MEASUREMENTS_ENTERED})\n"
                        f"- Model features available: {present_count}/{len(feature_columns)} (min {STRICT_MIN_PRESENT_FEATURES})\n\n"
                        "Enter more observations (or switch to Estimate mode)."
                    )
                    st.stop()

            prob = float(model.predict_proba(X_imp)[:, 1][0])
            band = risk_band(prob, thr_medium, thr_high)

            st.subheader("Result")
            st.write(band_badge(band))
            st.metric("Estimated mortality risk", f"{prob*100:.2f}%")
            st.caption(f"Model probability: {prob:.6f}")
            st.caption(f"Risk bands: LOW < {thr_medium:.2f}, MEDIUM {thr_medium:.2f}–{thr_high:.3f}, HIGH ≥ {thr_high:.3f}")

            # Missing UI: show specific missing model features (not just categories)
            st.subheader("What data was missing (and filled in by the model)")
            missing_pretty = format_missing_features(missing_features)

            if len(missing_pretty) == 0:
                st.write("None — all model features were available from your entries.")
            else:
                if strict_mode:
                    st.write(
                        "Some model inputs are missing. In Strict mode, the app only proceeds if enough data is present overall."
                    )
                else:
                    st.write("Missing inputs were filled using typical training values (Estimate mode).")

                for item in missing_pretty:
                    st.write(f"• {item}")

            # Explainability (always table)
            st.subheader("Why this risk estimate?")
            st.caption("Top factors that increased the model’s estimate. This is not proof of causation.")
            drivers_df = compute_top_drivers(bundle, X_imp, X_aligned, feature_columns, top_n=5)
            st.dataframe(drivers_df, use_container_width=True, hide_index=True)

            # Transparency: engineered values explanation (simple + clinician-friendly)
            with st.expander("How manual entries become model inputs", expanded=False):
                st.write(
                    "For each measurement, the model uses two engineered values:\n"
                    "- **Average**: mean of any values you entered across 12h / 6h / Now.\n"
                    "- **Latest**: the most recent value you entered (Now → 6h → 12h).\n\n"
                    "If you leave a measurement blank at all timepoints, it remains missing."
                )

            with st.expander("Engineered values available (pre-imputation)", expanded=False):
                row = pd.DataFrame([feats]).replace({pd.NA: np.nan})
                nonnull = row.loc[:, row.notna().any(axis=0)]
                st.dataframe(nonnull, use_container_width=True)

            with st.expander("What you did not enter (by timepoint)", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.write("**12h ago missing**")
                for x in sorted(missing_by_time["12h"]):
                    c1.write(f"• {x}")
                c2.write("**6h ago missing**")
                for x in sorted(missing_by_time["6h"]):
                    c2.write(f"• {x}")
                c3.write("**Now missing**")
                for x in sorted(missing_by_time["now"]):
                    c3.write(f"• {x}")

            with st.expander("Technical details", expanded=False):
                st.write(f"Bundle: {BUNDLE_PATH.as_posix()}")
                st.write(f"Features expected: {len(feature_columns)}")
                st.write(f"High threshold: {thr_high:.3f}")
                st.write(f"Medium threshold: {thr_medium:.2f}")

        except Exception as e:
            st.error("Something went wrong while predicting.")
            st.exception(e)
