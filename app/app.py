# app/app.py
"""
Streamlit demo app (educational).

- Upload a single patient .txt OR manually enter observations (12h / 6h / now)
- Produces calibrated probability + risk band
- Adds clinician-friendly missing item display, exports, and lightweight explainability

Notes:
- Requires outputs/model_bundle.joblib committed in the repo.
- Uses temp files for uploads and deletes them after parsing.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Path setup (robust imports)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your existing parsing + feature engineering
from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore


DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"

# "Medium" risk threshold is a demo threshold; high threshold comes from your bundle
DEFAULT_MEDIUM_THRESHOLD = 0.10

# Safety: if too little data entered, don't pretend we know.
MIN_VARS_FOR_CONFIDENT_PRED = 4  # variables with at least 1 observation (manual mode)


# -----------------------------
# Helpers: bundle + alignment
# -----------------------------
@st.cache_resource
def load_bundle(bundle_path: Path) -> dict:
    return joblib.load(bundle_path)


def safe_float(x: Any, default: float | None = None) -> float | None:
    """Convert to float safely; return default if not possible."""
    try:
        if x is None or x is pd.NA:
            return default
        return float(x)
    except Exception:
        return default


def align_and_impute(
    feats: dict,
    feature_columns: list[str],
    imputer,
) -> Tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    """
    Build a one-row dataframe in training feature order, report missing features,
    coerce to numeric, then impute.

    Returns: (X_imp, present_count, missing_features, X_aligned_pre_impute)
    """
    X = pd.DataFrame([feats])

    # Ensure all expected columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # Keep ONLY training columns, in order
    X = X[feature_columns]

    # Missing (pre-imputation)
    missing_mask = X.isna().iloc[0]
    missing_features = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    # Convert pd.NA -> np.nan; coerce to numeric; protect sklearn from NAType/object dtypes
    X = X.where(pd.notna(X), np.nan)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Impute
    X_imp_arr = imputer.transform(X)
    X_imp = pd.DataFrame(X_imp_arr, columns=feature_columns)

    return X_imp, present_count, missing_features, X


# -----------------------------
# Clinician-friendly labels
# -----------------------------
BASE_LABELS = {
    # Resp
    "RespRate": "Respiratory rate",
    "SpO2": "SpO₂",
    "SaO2": "SaO₂",
    "FiO2": "FiO₂ (fraction)",
    "PaO2": "PaO₂",
    "PaCO2": "PaCO₂",
    "pH": "pH",
    "HCO3": "HCO₃⁻",
    "MechVent": "Mechanical ventilation",
    # Cardiac
    "HeartRate": "Heart rate",
    "HR": "Heart rate",
    "MAP": "Mean arterial pressure (MAP)",
    "SysABP": "Systolic BP",
    "DiasABP": "Diastolic BP",
    "NIMAP": "Mean arterial pressure (NIBP)",
    "NISysABP": "Systolic BP (NIBP)",
    "NIDiasABP": "Diastolic BP (NIBP)",
    # Perfusion / labs
    "Lactate": "Lactate",
    "Glucose": "Glucose",
    "WBC": "White cell count",
    "Hct": "Haematocrit",
    "HCT": "Haematocrit",
    "Platelets": "Platelets",
    "BUN": "Urea (BUN)",
    "Urea": "Urea",
    "Creatinine": "Creatinine",
    "Sodium": "Sodium (Na)",
    "Na": "Sodium (Na)",
    "Potassium": "Potassium (K)",
    "K": "Potassium (K)",
    "Magnesium": "Magnesium (Mg)",
    "Mg": "Magnesium (Mg)",
}

SUFFIX_LABELS = {
    "mean": "Average of entered values",
    "last": "Most recent entered value",
    "was_measured": "Measured?",
    "prop_on": "Proportion of time on ventilation",
}

GROUPS = {
    "Cardiac": ["HR", "HeartRate", "MAP", "SysABP", "DiasABP", "NIMAP", "NISysABP", "NIDiasABP"],
    "Respiratory": ["RespRate", "SpO2", "SaO2", "FiO2", "MechVent"],
    "Blood gas": ["pH", "PaO2", "PaCO2", "HCO3"],
    "Bloods": ["Lactate", "Glucose", "WBC", "Platelets", "Urea", "BUN", "Creatinine", "Na", "K", "Mg", "Hct", "HCT", "Sodium", "Potassium", "Magnesium"],
}


def base_from_feature(feature_name: str) -> str:
    # handles e.g. "FiO2_last", "MechVent_prop_on", "RespRate_was_measured"
    parts = feature_name.split("_")
    if len(parts) == 1:
        return feature_name
    # prop_on is 2-part suffix
    if feature_name.endswith("_prop_on"):
        return feature_name[: -len("_prop_on")]
    # standard suffix
    return "_".join(parts[:-1])


def suffix_from_feature(feature_name: str) -> str:
    if feature_name.endswith("_prop_on"):
        return "prop_on"
    parts = feature_name.split("_")
    return parts[-1] if len(parts) > 1 else ""


def clinician_label(feature_name: str) -> str:
    b = base_from_feature(feature_name)
    s = suffix_from_feature(feature_name)

    base_label = BASE_LABELS.get(b, b)
    suffix_label = SUFFIX_LABELS.get(s, s)

    if s in ("mean", "last", "was_measured", "prop_on"):
        return f"{base_label} — {suffix_label}"
    return base_label


def group_for_base(base_var: str) -> str:
    for g, bases in GROUPS.items():
        if base_var in bases:
            return g
    # fallback: try label heuristics
    if base_var in ("RespRate", "SpO2", "SaO2", "FiO2", "MechVent"):
        return "Respiratory"
    return "Other"


def format_missing_grouped(missing_features: list[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for f in missing_features:
        b = base_from_feature(f)
        grp = group_for_base(b)
        grouped.setdefault(grp, []).append(clinician_label(f))
    # sort within groups
    for k in grouped:
        grouped[k] = sorted(list(dict.fromkeys(grouped[k])))
    return grouped


# -----------------------------
# Manual entry: input validation
# -----------------------------
@dataclass(frozen=True)
class VarSpec:
    label: str
    unit: str | None
    min_val: float | None
    max_val: float | None
    kind: str  # "numeric" | "binary" | "fraction"


VAR_SPECS: Dict[str, VarSpec] = {
    # Resp
    "RespRate": VarSpec("Respiratory rate", "breaths/min", 0, 80, "numeric"),
    "SpO2": VarSpec("SpO₂", "%", 0, 100, "numeric"),
    "SaO2": VarSpec("SaO₂", "%", 0, 100, "numeric"),
    "FiO2": VarSpec("FiO₂ (fraction)", None, 0.21, 1.0, "fraction"),
    "MechVent": VarSpec("Mechanical ventilation", None, None, None, "binary"),
    # Cardiac
    "HR": VarSpec("Heart rate", "bpm", 0, 250, "numeric"),
    "HeartRate": VarSpec("Heart rate", "bpm", 0, 250, "numeric"),
    "MAP": VarSpec("Mean arterial pressure (MAP)", "mmHg", 0, 200, "numeric"),
    "SysABP": VarSpec("Systolic BP", "mmHg", 0, 300, "numeric"),
    "DiasABP": VarSpec("Diastolic BP", "mmHg", 0, 200, "numeric"),
    "NISysABP": VarSpec("Systolic BP (NIBP)", "mmHg", 0, 300, "numeric"),
    "NIDiasABP": VarSpec("Diastolic BP (NIBP)", "mmHg", 0, 200, "numeric"),
    "NIMAP": VarSpec("Mean arterial pressure (NIBP)", "mmHg", 0, 200, "numeric"),
    # Blood gas
    "pH": VarSpec("pH", None, 6.8, 7.8, "numeric"),
    "PaO2": VarSpec("PaO₂", "mmHg", 0, 700, "numeric"),
    "PaCO2": VarSpec("PaCO₂", "mmHg", 0, 200, "numeric"),
    "HCO3": VarSpec("HCO₃⁻", "mmol/L", 0, 60, "numeric"),
    # Bloods
    "Lactate": VarSpec("Lactate", "mmol/L", 0, 30, "numeric"),
    "Glucose": VarSpec("Glucose", "mmol/L", 0, 60, "numeric"),
    "WBC": VarSpec("White cell count", "x10⁹/L", 0, 200, "numeric"),
    "Platelets": VarSpec("Platelets", "x10⁹/L", 0, 2000, "numeric"),
    "Urea": VarSpec("Urea", "mmol/L", 0, 80, "numeric"),
    "BUN": VarSpec("Urea (BUN)", "mg/dL", 0, 300, "numeric"),
    "Creatinine": VarSpec("Creatinine", "µmol/L", 0, 2000, "numeric"),
    "Na": VarSpec("Sodium (Na)", "mmol/L", 80, 200, "numeric"),
    "K": VarSpec("Potassium (K)", "mmol/L", 1.5, 10, "numeric"),
    "Mg": VarSpec("Magnesium (Mg)", "mmol/L", 0, 5, "numeric"),
    "Hct": VarSpec("Haematocrit", "%", 0, 80, "numeric"),
    "HCT": VarSpec("Haematocrit", "%", 0, 80, "numeric"),
}


def parse_numeric(text: str, spec: VarSpec, field_name: str) -> float:
    """
    Return np.nan if blank. If invalid, show clinician-friendly warning and return np.nan.
    """
    if text is None or str(text).strip() == "":
        return np.nan
    try:
        v = float(str(text).strip())
    except Exception:
        st.warning(f"⚠️ {field_name}: please enter a number{range_hint(spec)}. This field will be treated as missing.")
        return np.nan

    if spec.min_val is not None and v < spec.min_val:
        st.warning(f"⚠️ {field_name}: please enter a value between {spec.min_val:g} and {spec.max_val:g}. Treated as missing.")
        return np.nan
    if spec.max_val is not None and v > spec.max_val:
        st.warning(f"⚠️ {field_name}: please enter a value between {spec.min_val:g} and {spec.max_val:g}. Treated as missing.")
        return np.nan

    return v


def range_hint(spec: VarSpec) -> str:
    if spec.min_val is None or spec.max_val is None:
        return ""
    return f" ({spec.min_val:g}–{spec.max_val:g})"


def manual_engineer_features_from_observations(
    observations: Dict[str, Dict[str, Any]],
    starter_vars: List[str],
) -> Tuple[Dict[str, Any], int, List[str]]:
    """
    observations[var] = {"12h": value, "6h": value, "now": value}
    Returns (engineered_feature_dict, vars_entered_count, vars_entered_list)
    """
    feat: Dict[str, Any] = {}
    entered_vars: List[str] = []

    # order oldest->newest for "last": 12h, 6h, now
    time_keys = ["12h", "6h", "now"]

    for var in starter_vars:
        row = observations.get(var, {})
        vals = [row.get(k, np.nan) for k in time_keys]
        vals = [np.nan if (v is None or v is pd.NA) else v for v in vals]
        # numeric coercion already done during parse; binary -> 0/1 already
        s = pd.Series(vals, dtype="float64")

        has_any = bool(s.notna().any())
        feat[f"{var}_was_measured"] = 1 if has_any else 0

        if not has_any:
            feat[f"{var}_mean"] = np.nan
            feat[f"{var}_last"] = np.nan
            continue

        entered_vars.append(var)

        # mean of provided values (no inventions; only mean of actual entered values)
        feat[f"{var}_mean"] = float(s.mean(skipna=True))

        # last = most recent non-null in 12h->6h->now ordering
        last_val = np.nan
        for k in time_keys[::-1]:  # now, 6h, 12h
            v = row.get(k, np.nan)
            if v is None or v is pd.NA:
                v = np.nan
            if pd.notna(v):
                last_val = float(v)
                break
        feat[f"{var}_last"] = last_val

        # special handling for MechVent (binary)
        if var.lower() == "mechvent":
            mv = pd.to_numeric(s, errors="coerce")
            feat["MechVent_prop_on"] = float(mv.mean(skipna=True)) if mv.notna().any() else np.nan
            feat["MechVent_last"] = float(last_val) if pd.notna(last_val) else np.nan

    return feat, len(set(entered_vars)), sorted(list(set(entered_vars)))


# -----------------------------
# Explainability
# -----------------------------
def _unwrap_explain_model(model: Any) -> Any | None:
    """
    Prefer an uncalibrated estimator for SHAP if available.
    Works even if bundle only stores the calibrated wrapper.
    """
    # If the bundle explicitly stores raw model, prefer it
    # (you might add this later; safe to support now)
    if isinstance(model, dict) and "raw_model" in model:
        return model["raw_model"]

    # CalibratedClassifierCV: try to extract a fitted estimator
    try:
        from sklearn.calibration import CalibratedClassifierCV  # type: ignore

        if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_"):
            # pick the first fold's estimator (best available without bundle changes)
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, "estimator"):
                return cc.estimator
    except Exception:
        pass

    # If it's already a tree model, return as-is
    return model


def compute_top_risk_drivers(
    explain_model: Any,
    X_aligned_pre_impute: pd.DataFrame,
    feature_columns: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: [{"feature": "...", "label": "...", "value": ..., "push": ...}, ...]
    Only keeps factors that *push risk* (positive SHAP) when possible.
    Falls back safely if SHAP fails.
    """
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(explain_model)
        sv = explainer.shap_values(X_aligned_pre_impute)

        # handle binary outputs across SHAP versions
        if isinstance(sv, list):
            sv_pos = sv[1]
        else:
            sv_pos = sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv

        shap_vec = np.array(sv_pos[0], dtype=float).flatten()
        # positive contributions push risk upward
        idx_pos = np.where(shap_vec > 0)[0]
        if idx_pos.size > 0:
            order = idx_pos[np.argsort(np.abs(shap_vec[idx_pos]))[::-1]]
        else:
            # fallback: strongest absolute contributions if no positives detected
            order = np.argsort(np.abs(shap_vec))[::-1]

        rows: List[Dict[str, Any]] = []
        for i in order[:top_k]:
            fname = feature_columns[int(i)]
            raw_val = X_aligned_pre_impute.iloc[0, int(i)]
            rows.append(
                {
                    "feature": fname,
                    "label": clinician_label(fname),
                    "value": None if pd.isna(raw_val) else float(raw_val),
                    "push": float(shap_vec[int(i)]),
                }
            )
        # keep only positive push if we have them
        if idx_pos.size > 0:
            rows = [r for r in rows if r["push"] > 0][:top_k]
        return rows

    except Exception:
        # silent fail: explainability is nice-to-have, never block prediction
        return []


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo")
st.caption("For demonstration and education only — not a clinical decision tool.")

# Bundle load (repo-only)
bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error("Model bundle not found. Expected: outputs/model_bundle.joblib")
    st.stop()

bundle = load_bundle(bundle_path)

model = bundle["model"]
imputer = bundle["imputer"]
feature_columns: list[str] = bundle["feature_columns"]

thr_high = safe_float(bundle.get("default_threshold", 0.5), default=0.5) or 0.5
thr_medium = safe_float(bundle.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD), default=DEFAULT_MEDIUM_THRESHOLD) or DEFAULT_MEDIUM_THRESHOLD

starter_vars = bundle.get("starter_vars", None)
# If bundle doesn't store starter_vars, infer from feature columns (bases that have _mean/_last/_was_measured)
if starter_vars is None:
    bases = sorted({base_from_feature(c) for c in feature_columns})
    # keep only bases we have reasonable UI specs for (or default numeric)
    starter_vars = bases

# Clinician-friendly intro
with st.expander("What this tool does", expanded=False):
    st.write(
        """
This tool **estimates** a mortality risk score using a model trained on a public ICU dataset.
It combines available observations (vitals and blood results), converts them into model features,
and then produces a risk estimate and risk band.

It is designed as a **portfolio demonstration** of an end-to-end workflow — it is **not** validated for clinical use.
"""
    )

with st.expander("What data can I upload?", expanded=False):
    st.write(
        """
- Uploads should be a **single patient `.txt` file** in the same format as the dataset used in this project.
- Uploaded files are saved only to a **temporary location for parsing** and then deleted immediately after.
- The app does not need (and does not read) anything from your computer beyond the file you upload.
"""
    )

# Mode selector (keeps you on Manual after calculate)
mode = st.radio(
    "Choose input method",
    ["Upload patient file", "Manual entry"],
    horizontal=True,
    key="mode",
)

# Advanced / technical (moved to bottom later, but keep state here)
show_tech = st.toggle("Show technical details (for reviewers)", value=False)

st.divider()


def render_result_block(
    prob: float,
    band: str,
    thr_medium_val: float,
    thr_high_val: float,
):
    dot = {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}[band]
    st.subheader("Result")
    st.markdown(f"### {dot} {band} RISK")
    st.metric("Estimated mortality risk", f"{prob*100:.2f}%")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Model probability: {prob:.6f}")
    st.caption(f"Risk bands: LOW < {thr_medium_val:.2f}, MEDIUM {thr_medium_val:.2f}–{thr_high_val:.3f}, HIGH ≥ {thr_high_val:.3f}")


def risk_band(prob: float, thr_medium_val: float, thr_high_val: float) -> str:
    if prob < thr_medium_val:
        return "LOW"
    if prob < thr_high_val:
        return "MEDIUM"
    return "HIGH"


def export_buttons(payload: dict, filename_prefix: str = "icu_mortality_demo"):
    # JSON
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    st.download_button(
        "Download summary (JSON)",
        data=js.encode("utf-8"),
        file_name=f"{filename_prefix}_summary.json",
        mime="application/json",
        use_container_width=True,
    )

    # CSV (one row)
    flat = {
        "timestamp_utc": payload.get("timestamp_utc"),
        "mode": payload.get("mode"),
        "patient_file": payload.get("patient_file"),
        "probability": payload.get("probability"),
        "risk_band": payload.get("risk_band"),
        "thr_medium": payload.get("thr_medium"),
        "thr_high": payload.get("thr_high"),
        "variables_entered": "; ".join(payload.get("variables_entered", [])),
        "missing_items": "; ".join(payload.get("missing_items", [])),
        "top_drivers": "; ".join([f"{d.get('label')} (push {d.get('push'):.3f})" for d in payload.get("top_risk_drivers", [])]),
    }
    csv_buf = io.StringIO()
    pd.DataFrame([flat]).to_csv(csv_buf, index=False)
    st.download_button(
        "Download summary (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"{filename_prefix}_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------
# Upload mode
# -----------------------------
if mode == "Upload patient file":
    uploaded = st.file_uploader("Upload a single patient .txt file", type=["txt"])

    if uploaded is None:
        st.info("Upload a patient file to get a prediction.")
        # Technical details at bottom
        if show_tech:
            st.divider()
            with st.expander("Technical details", expanded=False):
                st.write("Bundled model: outputs/model_bundle.joblib")
                st.write(f"Features expected: {len(feature_columns)}")
        st.stop()

    # safety: limit size
    if uploaded.size > 2_000_000:  # 2MB
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    # Temp file -> parse -> delete
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_features, X_aligned = align_and_impute(feats, feature_columns, imputer)
        prob = float(model.predict_proba(X_imp)[:, 1][0])
        band = risk_band(prob, thr_medium, thr_high)

        render_result_block(prob, band, thr_medium, thr_high)

        # Missing items (clinician-friendly)
        st.subheader("What data was missing (and filled in by the model)")
        grouped_missing = format_missing_grouped(missing_features)
        if len(missing_features) == 0:
            st.write("No missing items detected for this patient file.")
        else:
            for grp, items in grouped_missing.items():
                with st.expander(grp, expanded=False):
                    for it in items:
                        st.write(f"• {it}")

        # Explainability (for upload too)
        st.subheader("Why this risk estimate?")
        st.write("These are the **top factors** that increased the model’s estimate today. This is **not** proof of causation.")

        explain_model = _unwrap_explain_model(model)
        show_explain = (band in ("MEDIUM", "HIGH"))  # guardrail: compute by default only when it matters
        if show_explain:
            drivers = compute_top_risk_drivers(explain_model, X_aligned, feature_columns, top_k=5) if explain_model is not None else []
        else:
            drivers = []

        if len(drivers) == 0:
            st.caption("No explanation available for this case (or explanation could not be generated).")
        else:
            df_drv = pd.DataFrame(
                [
                    {
                        "Factor": d["label"],
                        "Value used": "" if d["value"] is None else f"{d['value']:.3f}",
                        "Pushes risk by": f"{d['push']:.3f}",
                    }
                    for d in drivers
                ]
            )
            st.dataframe(df_drv, hide_index=True, use_container_width=True)

        # Exports
        st.subheader("Export")
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "upload",
            "patient_file": uploaded.name,
            "probability": prob,
            "risk_band": band,
            "thr_medium": thr_medium,
            "thr_high": thr_high,
            "variables_entered": [],  # upload mode: unknown from file at UI layer
            "missing_items": [clinician_label(m) for m in missing_features],
            "top_risk_drivers": drivers,
        }
        export_buttons(payload, filename_prefix=Path(uploaded.name).stem)

        # Technical details at bottom
        if show_tech:
            st.divider()
            with st.expander("Technical details", expanded=False):
                st.write("Bundled model: outputs/model_bundle.joblib")
                st.write(f"Features expected: {len(feature_columns)}")
                st.write(f"Features present: {present_count}/{len(feature_columns)}")
                st.write(f"Missing features (imputed): {len(missing_features)}")
                with st.expander("Engineered values (one row, pre-imputation)", expanded=False):
                    st.dataframe(X_aligned, use_container_width=True)

    except Exception as e:
        st.error("Something went wrong while predicting.")
        # Keep clinician-facing error simple; full trace only in reviewer mode
        if show_tech:
            st.exception(e)
        else:
            st.info("Please try again, or use Manual entry if you want to input observations directly.")

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -----------------------------
# Manual mode
# -----------------------------
else:
    st.subheader("Manual entry")
    st.caption("Enter observations at up to three timepoints. Leave blank if unknown.")

    # Derive a reasonable list of “input bases” from starter_vars and what we can label
    base_vars = list(dict.fromkeys([str(v) for v in starter_vars]))  # preserve order, unique

    # Quick grouping order (pH first in blood gas, BP inside cardiac)
    def sort_key(v: str) -> Tuple[int, int]:
        grp = group_for_base(v)
        grp_order = {"Cardiac": 0, "Respiratory": 1, "Blood gas": 2, "Bloods": 3, "Other": 9}.get(grp, 9)
        # within blood gas: pH first
        if grp == "Blood gas":
            inner = {"pH": 0, "PaO2": 1, "PaCO2": 2, "HCO3": 3}.get(v, 9)
        else:
            inner = 9
        return (grp_order, inner)

    base_vars_sorted = sorted(base_vars, key=sort_key)

    # Session state for quick-fill and persist
    if "manual_inputs" not in st.session_state:
        st.session_state["manual_inputs"] = {}

    def set_all_from_now():
        for var in base_vars_sorted:
            d = st.session_state["manual_inputs"].get(var, {})
            now = d.get("now", "")
            if now not in (None, ""):
                d["6h"] = now
                d["12h"] = now
            st.session_state["manual_inputs"][var] = d

    def set_6h_from_now():
        for var in base_vars_sorted:
            d = st.session_state["manual_inputs"].get(var, {})
            now = d.get("now", "")
            if now not in (None, ""):
                d["6h"] = now
            st.session_state["manual_inputs"][var] = d

    def clear_all():
        st.session_state["manual_inputs"] = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Copy NOW → 6h & 12h", on_click=set_all_from_now, use_container_width=True)
    with c2:
        st.button("Copy NOW → 6h", on_click=set_6h_from_now, use_container_width=True)
    with c3:
        st.button("Clear all", on_click=clear_all, use_container_width=True)

    st.divider()

    def render_group(group_name: str, vars_in_group: List[str]):
        with st.expander(group_name, expanded=True):
            # Header row
            h1, h2, h3, h4 = st.columns([2.3, 1.2, 1.2, 1.2])
            h1.write("**Measurement**")
            h2.write("**12h ago**")
            h3.write("**6h ago**")
            h4.write("**Now**")

            for var in vars_in_group:
                spec = VAR_SPECS.get(var, VarSpec(BASE_LABELS.get(var, var), None, None, None, "numeric"))

                row = st.session_state["manual_inputs"].get(var, {})
                # defaults
                row.setdefault("12h", "")
                row.setdefault("6h", "")
                row.setdefault("now", "")
                st.session_state["manual_inputs"][var] = row

                col_label, col_12, col_6, col_now = st.columns([2.3, 1.2, 1.2, 1.2])

                label = spec.label
                if spec.unit:
                    label = f"{label} ({spec.unit})"
                if spec.kind == "fraction" and var == "FiO2":
                    label = f"{label} — enter as fraction (0.21–1.00)"

                col_label.write(label)

                if spec.kind == "binary":
                    # Binary selector with blank
                    opts = ["", "No", "Yes"]
                    v12 = col_12.selectbox("", opts, index=opts.index(row["12h"]) if row["12h"] in opts else 0, key=f"{var}_12h")
                    v6 = col_6.selectbox("", opts, index=opts.index(row["6h"]) if row["6h"] in opts else 0, key=f"{var}_6h")
                    vnow = col_now.selectbox("", opts, index=opts.index(row["now"]) if row["now"] in opts else 0, key=f"{var}_now")
                    st.session_state["manual_inputs"][var] = {"12h": v12, "6h": v6, "now": vnow}
                else:
                    # Numeric text inputs (blank allowed)
                    ph = ""  # keep placeholders consistent: none
                    v12 = col_12.text_input("", value=row["12h"], placeholder=ph, key=f"{var}_12h")
                    v6 = col_6.text_input("", value=row["6h"], placeholder=ph, key=f"{var}_6h")
                    vnow = col_now.text_input("", value=row["now"], placeholder=ph, key=f"{var}_now")
                    st.session_state["manual_inputs"][var] = {"12h": v12, "6h": v6, "now": vnow}

    # Build groups from what exists in model bases
    grouped_vars: Dict[str, List[str]] = {"Cardiac": [], "Respiratory": [], "Blood gas": [], "Bloods": [], "Other": []}
    for v in base_vars_sorted:
        grouped_vars.setdefault(group_for_base(v), []).append(v)

    # Ensure BP lives in Cardiac (explicit)
    for bp in ["MAP", "SysABP", "DiasABP", "NIMAP", "NISysABP", "NIDiasABP"]:
        if bp in base_vars_sorted and bp not in grouped_vars["Cardiac"]:
            grouped_vars["Cardiac"].append(bp)

    # Blood gas order: pH first
    if "Blood gas" in grouped_vars:
        bg = grouped_vars["Blood gas"]
        order = ["pH", "PaO2", "PaCO2", "HCO3"]
        grouped_vars["Blood gas"] = [x for x in order if x in bg] + [x for x in bg if x not in order]

    for grp_name in ["Cardiac", "Respiratory", "Blood gas", "Bloods", "Other"]:
        if grouped_vars.get(grp_name):
            render_group(grp_name, grouped_vars[grp_name])

    st.divider()

    # Calculate
    calc = st.button("Calculate risk", type="primary", use_container_width=True)

    if calc:
        # Parse observations into numeric/binary
        observations: Dict[str, Dict[str, Any]] = {}

        for var in base_vars_sorted:
            spec = VAR_SPECS.get(var, VarSpec(BASE_LABELS.get(var, var), None, None, None, "numeric"))
            row = st.session_state["manual_inputs"].get(var, {"12h": "", "6h": "", "now": ""})

            if spec.kind == "binary":
                def bin_to_num(x: str) -> float:
                    if x == "Yes":
                        return 1.0
                    if x == "No":
                        return 0.0
                    return np.nan

                observations[var] = {
                    "12h": bin_to_num(row.get("12h", "")),
                    "6h": bin_to_num(row.get("6h", "")),
                    "now": bin_to_num(row.get("now", "")),
                }
            else:
                observations[var] = {
                    "12h": parse_numeric(row.get("12h", ""), spec, f"{spec.label} (12h ago)"),
                    "6h": parse_numeric(row.get("6h", ""), spec, f"{spec.label} (6h ago)"),
                    "now": parse_numeric(row.get("now", ""), spec, f"{spec.label} (now)"),
                }

        # Engineer features only from what was actually entered (no made-up means)
        engineered, vars_entered_count, vars_entered_list = manual_engineer_features_from_observations(
            observations=observations,
            starter_vars=list(base_vars_sorted),
        )

        # Data sufficiency gate
        st.subheader("Data check")
        st.write(f"Variables entered: **{vars_entered_count}**")

        estimate_anyway = False
        if vars_entered_count < MIN_VARS_FOR_CONFIDENT_PRED:
            st.warning(
                f"Not enough data entered to give a reliable estimate yet "
                f"(recommend at least {MIN_VARS_FOR_CONFIDENT_PRED} variables)."
            )
            estimate_anyway = st.checkbox("Estimate anyway (may be less reliable)", value=False)

            if not estimate_anyway:
                st.stop()

        try:
            X_imp, present_count, missing_features, X_aligned = align_and_impute(engineered, feature_columns, imputer)
            prob = float(model.predict_proba(X_imp)[:, 1][0])
            band = risk_band(prob, thr_medium, thr_high)

            render_result_block(prob, band, thr_medium, thr_high)

            # Show ONLY the selected band explicitly (already done above),
            # and avoid extra confusing binary wording.

            # Missing items (clinician-friendly)
            st.subheader("What data was missing (and filled in by the model)")
            grouped_missing = format_missing_grouped(missing_features)
            if len(missing_features) == 0:
                st.write("No missing items detected from your entries.")
            else:
                for grp, items in grouped_missing.items():
                    with st.expander(grp, expanded=False):
                        for it in items:
                            st.write(f"• {it}")

            # Explainability (always present for manual; computed when useful)
            st.subheader("Why this risk estimate?")
            st.write("These are the **top factors** that increased the model’s estimate today. This is **not** proof of causation.")

            explain_model = _unwrap_explain_model(model)
            drivers: List[Dict[str, Any]] = []

            # Performance guardrail: generate by default for medium/high; for low keep lightweight
            if band in ("MEDIUM", "HIGH") and explain_model is not None:
                drivers = compute_top_risk_drivers(explain_model, X_aligned, feature_columns, top_k=5)

            if len(drivers) == 0:
                st.caption("No explanation available for this case (or explanation could not be generated).")
            else:
                df_drv = pd.DataFrame(
                    [
                        {
                            "Factor": d["label"],
                            "Value used": "" if d["value"] is None else f"{d['value']:.3f}",
                            "Pushes risk by": f"{d['push']:.3f}",
                        }
                        for d in drivers
                    ]
                )
                st.dataframe(df_drv, hide_index=True, use_container_width=True)

            # Exports
            st.subheader("Export")
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "manual",
                "patient_file": None,
                "probability": prob,
                "risk_band": band,
                "thr_medium": thr_medium,
                "thr_high": thr_high,
                "variables_entered": [BASE_LABELS.get(v, v) for v in vars_entered_list],
                "missing_items": [clinician_label(m) for m in missing_features],
                "top_risk_drivers": drivers,
            }
            export_buttons(payload, filename_prefix="manual_entry")

            if show_tech:
                st.divider()
                with st.expander("Technical details", expanded=False):
                    st.write(f"Features expected: {len(feature_columns)}")
                    st.write(f"Features present: {present_count}/{len(feature_columns)}")
                    st.write(f"Missing features (imputed): {len(missing_features)}")
                    with st.expander("Engineered values (one row, pre-imputation)", expanded=False):
                        # show only non-null engineered values to reduce the wall of None/NaN
                        non_null = X_aligned.loc[:, X_aligned.notna().iloc[0]]
                        st.dataframe(non_null, use_container_width=True)

        except Exception as e:
            st.error("Something went wrong while predicting.")
            if show_tech:
                st.exception(e)
            else:
                st.info("Please review your entries and try again. If the issue persists, use Upload mode.")


# -----------------------------
# Bottom technical details (always last)
# -----------------------------
st.divider()
if show_tech:
    with st.expander("Environment & versions (reviewers)", expanded=False):
        import sklearn  # type: ignore

        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"scikit-learn: {sklearn.__version__}")
        st.write(f"numpy: {np.__version__}")
        st.write(f"pandas: {pd.__version__}")
        st.write(f"joblib: {joblib.__version__}")
        st.write("Bundled model: outputs/model_bundle.joblib")
