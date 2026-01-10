# app/app.py
"""
ICU Mortality Risk Demo (Portfolio App)

For demonstration + education only — NOT a clinical decision tool.

This app supports:
1) Uploading a single patient .txt file (public dataset format)
2) Manual entry of observations at up to 3 timepoints (12h / 6h / now)

It loads a bundled model at: outputs/model_bundle.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Robust imports (Streamlit Cloud safe)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.step_01_load_raw import load_patient_long  # type: ignore
from scripts.step_02_batch_features import summarise_patient  # type: ignore

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "outputs" / "model_bundle.joblib"

# Default thresholds (bundle can override)
DEFAULT_THR_MEDIUM = 0.10

# Strict mode rules (tuned for usability, not “mathematical truth”)
STRICT_MIN_ENGINEERED_PRESENT_FRAC = 0.40   # require at least 40% of model inputs present pre-imputation
STRICT_MIN_OBSERVATIONS_ENTERED = 6         # require at least 6 raw observations entered (manual), or parsed (upload)

# -----------------------------
# Display mapping helpers
# -----------------------------
@dataclass(frozen=True)
class FieldSpec:
    key_base: str
    label: str
    unit: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    kind: str = "float"  # "float" or "bool" or "int"


# These are “clinician-facing” manual inputs.
# They map onto engineered model features like {Var}_mean and {Var}_last.
MANUAL_FIELDS = {
    "Cardiac": [
        FieldSpec("HeartRate", "Heart rate", "bpm", 0, 250, "int"),
        FieldSpec("NIMAP", "Mean arterial pressure (NIBP)", "mmHg", 0, 200, "float"),
        FieldSpec("NISysABP", "Systolic BP (NIBP)", "mmHg", 0, 300, "float"),
        FieldSpec("NIDiasABP", "Diastolic BP (NIBP)", "mmHg", 0, 200, "float"),
    ],
    "Respiratory": [
        FieldSpec("RespRate", "Respiratory rate", "breaths/min", 0, 80, "int"),
        FieldSpec("SaO2", "SaO₂", "%", 0, 100, "float"),
        FieldSpec("FiO2", "FiO₂ (fraction)", "fraction (0.21–1.00)", 0.21, 1.0, "float"),
        FieldSpec("MechVent", "Mechanical ventilation", "Yes/No", kind="bool"),
    ],
    "Blood gas": [
        FieldSpec("pH", "pH", "", 6.8, 7.8, "float"),
        FieldSpec("HCO3", "HCO₃⁻", "mmol/L", 0, 60, "float"),
        FieldSpec("Lactate", "Lactate", "mmol/L", 0, 30, "float"),
    ],
    "Bloods": [
        FieldSpec("Na", "Sodium (Na)", "mmol/L", 90, 200, "float"),
        FieldSpec("K", "Potassium (K)", "mmol/L", 1, 10, "float"),
        FieldSpec("Mg", "Magnesium (Mg)", "mmol/L", 0, 5, "float"),
        FieldSpec("Creatinine", "Creatinine", "µmol/L", 0, 2000, "float"),
        FieldSpec("BUN", "Urea (BUN)", "mg/dL", 0, 200, "float"),
        FieldSpec("Platelets", "Platelets", "x10⁹/L", 0, 1500, "float"),
        FieldSpec("HCT", "Haematocrit", "%", 0, 80, "float"),
        FieldSpec("Glucose", "Glucose", "mmol/L", 0, 60, "float"),
    ],
    "Other": [
        FieldSpec("GCS", "GCS", "", 3, 15, "int"),
    ],
}

# Map model feature names to clinician-readable labels for reporting missing/driver features.
BASE_LABELS = {fs.key_base: (fs.label, fs.unit) for group in MANUAL_FIELDS.values() for fs in group}


def pretty_model_feature_name(col: str) -> str:
    """
    Convert model feature like 'HeartRate_mean' to 'Heart rate — average'
    and try to attach units where known.
    """
    suffix = ""
    base = col

    if col.endswith("_mean"):
        base = col[:-5]
        suffix = " — average"
    elif col.endswith("_last"):
        base = col[:-5]
        suffix = " — latest"
    elif col.endswith("_was_measured"):
        base = col[:-13]
        suffix = " — measured?"
    elif col.startswith("MechVent_") and col in ("MechVent_prop_on",):
        base = "MechVent"
        suffix = " — proportion on"
    else:
        base = col

    if base in BASE_LABELS:
        label, unit = BASE_LABELS[base]
        if unit and suffix not in (" — measured?",):
            return f"{label} ({unit}){suffix}".strip()
        return f"{label}{suffix}".strip()

    cleaned = base.replace("_", " ")
    return f"{cleaned}{suffix}".strip()


def feature_category(col: str) -> str:
    base = col
    for sfx in ("_mean", "_last", "_was_measured"):
        if base.endswith(sfx):
            base = base[: -len(sfx)]
    if base in BASE_LABELS:
        for section, fields in MANUAL_FIELDS.items():
            if any(f.key_base == base for f in fields):
                return section
    return "Other model inputs"


# -----------------------------
# Model/bundle utilities
# -----------------------------
@st.cache_resource
def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def risk_band(prob: float, thr_medium: float, thr_high: float) -> str:
    if prob < thr_medium:
        return "LOW"
    if prob < thr_high:
        return "MEDIUM"
    return "HIGH"


def band_dot(band: str) -> str:
    return {"LOW": "🟢", "MEDIUM": "🟠", "HIGH": "🔴"}.get(band, "⚪")


def reliability_label(present_frac: float) -> tuple[str, str]:
    if present_frac >= 0.70:
        return "High", "🟢"
    if present_frac >= 0.40:
        return "Moderate", "🟠"
    return "Low", "🔴"


def align_and_impute(
    feats: dict,
    feature_columns: list[str],
    imputer: Any
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    X = pd.DataFrame([feats])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_columns]
    X = X.where(pd.notna(X), np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    missing_mask = X.isna().iloc[0]
    missing_cols = X.columns[missing_mask].tolist()
    present_count = int((~missing_mask).sum())

    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    return X_imp, present_count, missing_cols, X


def grouped_missing(missing_cols: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for col in missing_cols:
        cat = feature_category(col)
        out.setdefault(cat, []).append(pretty_model_feature_name(col))
    for k in out:
        out[k] = sorted(out[k])
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


# -----------------------------
# Explainability (best-effort)
# -----------------------------
def _unwrap_calibrated_estimator(model: Any) -> Any:
    try:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, "estimator"):
                return cc.estimator
        if hasattr(model, "estimator"):
            return model.estimator
    except Exception:
        pass
    return model


@st.cache_resource
def _get_shap_explainer(_model: Any):
    """
    IMPORTANT: argument name starts with underscore so Streamlit will NOT hash it.
    """
    try:
        import shap  # type: ignore
        base = _unwrap_calibrated_estimator(_model)
        return shap.TreeExplainer(base)
    except Exception:
        return None


def explain_top_drivers(
    model: Any,
    X_imp: pd.DataFrame,
    X_aligned: pd.DataFrame,
    feature_columns: list[str],
    top_k: int = 5
) -> pd.DataFrame:
    explainer = _get_shap_explainer(model)
    if explainer is not None:
        try:
            sv = explainer.shap_values(X_imp)

            if isinstance(sv, list) and len(sv) >= 2:
                sv_pos = sv[1]
            else:
                sv_arr = np.asarray(sv)
                if sv_arr.ndim == 3:
                    sv_pos = sv_arr[:, :, 1]
                else:
                    sv_pos = sv_arr

            shap_row = np.asarray(sv_pos)[0].astype(float)
            cols = np.array(feature_columns)

            pos_idx = np.where(shap_row > 0)[0]
            if pos_idx.size > 0:
                order = pos_idx[np.argsort(shap_row[pos_idx])[::-1]][:top_k]
                data = []
                for i in order:
                    col = cols[i]
                    value = float(X_imp.iloc[0, i])
                    was_missing = bool(pd.isna(X_aligned.iloc[0, i]))
                    data.append(
                        {
                            "What increased risk": pretty_model_feature_name(col),
                            "Value used": value,
                            "Influence on estimate": float(shap_row[i]),
                            "Entered by user?": "No (filled)" if was_missing else "Yes",
                        }
                    )
                return pd.DataFrame(data)

        except Exception:
            pass

    base = _unwrap_calibrated_estimator(model)
    if hasattr(base, "feature_importances_"):
        try:
            imp = np.asarray(base.feature_importances_, dtype=float)
            cols = np.array(feature_columns)

            present_mask = ~pd.isna(X_aligned.iloc[0].to_numpy())
            idx = np.where(present_mask)[0]
            if idx.size == 0:
                return pd.DataFrame(columns=["What increased risk", "Value used", "Influence on estimate", "Entered by user?"])

            order = idx[np.argsort(imp[idx])[::-1]][:top_k]
            data = []
            for i in order:
                col = cols[i]
                value = float(X_imp.iloc[0, i])
                data.append(
                    {
                        "What increased risk": pretty_model_feature_name(col),
                        "Value used": value,
                        "Influence on estimate": float(imp[i]),
                        "Entered by user?": "Yes",
                    }
                )
            return pd.DataFrame(data)
        except Exception:
            pass

    return pd.DataFrame(columns=["What increased risk", "Value used", "Influence on estimate", "Entered by user?"])


# -----------------------------
# Manual entry parsing/engineering
# -----------------------------
def _parse_number(text: str, spec: FieldSpec) -> tuple[Optional[float], Optional[str]]:
    t = (text or "").strip()
    if t == "":
        return None, None

    if spec.kind == "bool":
        if t.lower() in ("yes", "y", "true", "1"):
            return 1.0, None
        if t.lower() in ("no", "n", "false", "0"):
            return 0.0, None
        return None, "Please enter Yes or No."

    try:
        v = float(t)
        if spec.kind == "int":
            v = float(int(round(v)))
    except Exception:
        return None, "Please enter a number."

    if spec.min_val is not None and v < spec.min_val:
        return None, f"Please enter a value between {spec.min_val:g} and {spec.max_val:g}."
    if spec.max_val is not None and v > spec.max_val:
        return None, f"Please enter a value between {spec.min_val:g} and {spec.max_val:g}."

    return v, None


def engineer_from_timepoints(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) == 0:
        return None, None
    mean = float(np.mean(vals))
    last = float(vals[-1])
    return mean, last


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ICU Mortality Risk Demo", layout="centered")

st.title("ICU Mortality Risk Demo")
st.caption("For demonstration and education only — not a clinical decision tool.")

with st.expander("What this tool does", expanded=False):
    st.write(
        "This demo estimates mortality risk using a model trained on a **public ICU dataset**. "
        "It converts observations into inputs the model understands, then produces a risk estimate and risk band. "
        "If information is missing, the app can (optionally) use **typical values** from training data to complete the calculation."
    )

with st.expander("What data can I upload?", expanded=False):
    st.write(
        "Upload a single patient `.txt` file in the **same public dataset format** used by this project. "
        "Alternatively, use Manual entry to enter observations directly."
    )

bundle_path = DEFAULT_BUNDLE_PATH
if not bundle_path.exists():
    st.error(f"Model bundle not found at: {bundle_path.as_posix()}")
    st.stop()

bundle = load_bundle(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
feature_columns = bundle["feature_columns"]

thr_high = float(bundle.get("default_threshold", 0.5))
thr_medium = float(bundle.get("medium_threshold", DEFAULT_THR_MEDIUM))

st.subheader("Choose input method")
input_method = st.radio(
    label="",
    options=["Upload patient file", "Manual entry"],
    horizontal=True,
)

mode = st.radio(
    "Prediction mode",
    options=["Strict (more transparent)", "Estimate (uses typical values for missing inputs)"],
    index=0,
)
st.caption(
    "• **Strict:** only gives a result if there is enough real data to be meaningful.\n"
    "• **Estimate:** can still give a result with limited data, but may rely on typical values."
)


def strict_gate_ok(engineered_present_count: int, observations_entered: int) -> tuple[bool, str]:
    total = len(feature_columns)
    frac = engineered_present_count / max(total, 1)
    if observations_entered < STRICT_MIN_OBSERVATIONS_ENTERED:
        return False, f"Not enough observations entered (need at least {STRICT_MIN_OBSERVATIONS_ENTERED})."
    if frac < STRICT_MIN_ENGINEERED_PRESENT_FRAC:
        need = int(np.ceil(STRICT_MIN_ENGINEERED_PRESENT_FRAC * total))
        return False, f"Not enough model inputs available (need about {need} of {total})."
    return True, ""


def render_result(
    container: Any,
    prob: float,
    present_count: int,
    missing_cols: list[str],
    X_imp: pd.DataFrame,
    X_aligned: pd.DataFrame,
    observations_entered: int,
    engineered_present_count: int,
):
    total = len(feature_columns)
    present_frac = present_count / max(total, 1)
    band = risk_band(prob, thr_medium, thr_high)
    rel_text, rel_dot = reliability_label(present_frac)

    with container:
        st.subheader("Result")
        st.markdown(f"### {band_dot(band)} {band} RISK")
        st.metric("Estimated mortality risk", f"{prob*100:.2f}%")
        st.caption(f"Model probability: {prob:.6f}")

        st.write(
            f"**Reliability:** {rel_dot} **{rel_text}**  "
            f"(model inputs present: {engineered_present_count}/{total}; "
            f"observations entered: {observations_entered})"
        )

        st.caption(
            f"Risk bands: LOW < {thr_medium:.2f}, MEDIUM {thr_medium:.2f}–{thr_high:.3f}, HIGH ≥ {thr_high:.3f}"
        )

        if missing_cols:
            st.subheader("What data was missing (and filled using typical values)")
            st.write(
                "Some model inputs were missing. "
                "**Typical values were used for missing items (from the training dataset).** "
                "These are **not from your patient file**."
            )

            grouped = grouped_missing(missing_cols)
            counts_line = " • ".join([f"{k}: {len(v)}" for k, v in grouped.items()])
            st.caption(f"Missing & filled: {len(missing_cols)} inputs ({counts_line})")

            for cat, items in grouped.items():
                st.markdown(f"**{cat}**")
                for it in items:
                    st.write(f"• {it}")

        st.subheader("Why this risk estimate?")
        st.caption("Top factors that increased the model’s estimate. This is not proof of causation.")

        drivers = explain_top_drivers(model, X_imp, X_aligned, feature_columns, top_k=5)
        if drivers.empty:
            st.info("No clear top drivers were available for display with the current data.")
        else:
            display = drivers.copy()
            if "Value used" in display.columns:
                display["Value used"] = display["Value used"].map(
                    lambda x: f"{x:.3g}" if isinstance(x, (int, float, np.floating)) else str(x)
                )
            if "Influence on estimate" in display.columns:
                display["Influence on estimate"] = display["Influence on estimate"].map(
                    lambda x: f"{x:.3g}" if isinstance(x, (int, float, np.floating)) else str(x)
                )
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("Export")
        export_df = pd.DataFrame(
            {
                "probability": [prob],
                "risk_band": [band],
                "inputs_present": [engineered_present_count],
                "inputs_total": [total],
                "observations_entered": [observations_entered],
                "missing_inputs_count": [len(missing_cols)],
            }
        )
        st.download_button(
            "Download result summary (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="icu_mortality_demo_result.csv",
            mime="text/csv",
        )

        with st.expander("Technical details", expanded=False):
            st.write(f"Bundle path: `{bundle_path.as_posix()}`")


# -----------------------------
# Upload patient file mode
# -----------------------------
if input_method == "Upload patient file":
    st.subheader("Upload patient file")
    uploaded = st.file_uploader("Upload a single patient `.txt` file", type=["txt"])

    # Results should appear BELOW the uploader
    upload_result_container = st.container()

    if uploaded is None:
        st.info("Upload a patient file to get a prediction.")
        st.stop()

    if uploaded.size > 2_000_000:
        st.error("File too large for this demo (max 2MB).")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        long_df = load_patient_long(tmp_path)
        feats = summarise_patient(long_df)

        X_imp, present_count, missing_cols, X_aligned = align_and_impute(feats, feature_columns, imputer)

        engineered_present_count = present_count
        observations_entered = int(engineered_present_count)

        if mode.startswith("Strict"):
            ok, reason = strict_gate_ok(engineered_present_count, observations_entered)
            if not ok:
                with upload_result_container:
                    st.warning(
                        "Strict mode: insufficient data to provide a meaningful estimate.\n\n"
                        f"Reason: {reason}\n\n"
                        "Switch to **Estimate mode** if you still want an approximate result using typical values."
                    )
                st.stop()

        prob = float(model.predict_proba(X_imp)[:, 1][0])

        # Persist so it doesn't vanish on rerun
        st.session_state["upload_result"] = dict(
            prob=prob,
            present_count=present_count,
            missing_cols=missing_cols,
            X_imp=X_imp,
            X_aligned=X_aligned,
            observations_entered=observations_entered,
            engineered_present_count=engineered_present_count,
        )

        res = st.session_state.get("upload_result")
        if res:
            render_result(
                container=upload_result_container,
                prob=res["prob"],
                present_count=res["present_count"],
                missing_cols=res["missing_cols"],
                X_imp=res["X_imp"],
                X_aligned=res["X_aligned"],
                observations_entered=res["observations_entered"],
                engineered_present_count=res["engineered_present_count"],
            )

    except Exception as e:
        with upload_result_container:
            st.error("Something went wrong while predicting.")
            st.exception(e)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -----------------------------
# Manual entry mode
# -----------------------------
if input_method == "Manual entry":
    st.subheader("Manual entry")
    st.caption("Enter observations at up to three timepoints. Leave blank if unknown.")

    raw_obs: dict[str, list[Optional[float]]] = {}
    raw_errors: list[str] = []
    observations_entered = 0

    for section, fields in MANUAL_FIELDS.items():
        st.markdown(f"### {section}")
        hdr = st.columns([3, 2, 2, 2])
        hdr[0].write("**Measurement**")
        hdr[1].write("**12h ago**")
        hdr[2].write("**6h ago**")
        hdr[3].write("**Now**")

        for spec in fields:
            row = st.columns([3, 2, 2, 2])

            label = spec.label
            if spec.unit:
                label = f"{label} ({spec.unit})"
            row[0].write(label)

            vals: list[Optional[float]] = []

            if spec.kind == "bool":
                opts = ["", "No", "Yes"]
                v12 = row[1].selectbox("", opts, key=f"{spec.key_base}_12", label_visibility="collapsed")
                v6 = row[2].selectbox("", opts, key=f"{spec.key_base}_6", label_visibility="collapsed")
                vnow = row[3].selectbox("", opts, key=f"{spec.key_base}_0", label_visibility="collapsed")

                for v in (v12, v6, vnow):
                    if v == "":
                        vals.append(None)
                    else:
                        observations_entered += 1
                        vals.append(1.0 if v == "Yes" else 0.0)
            else:
                t12 = row[1].text_input("", value="", key=f"{spec.key_base}_12", label_visibility="collapsed")
                t6 = row[2].text_input("", value="", key=f"{spec.key_base}_6", label_visibility="collapsed")
                tnow = row[3].text_input("", value="", key=f"{spec.key_base}_0", label_visibility="collapsed")

                for t in (t12, t6, tnow):
                    v, err = _parse_number(t, spec)
                    if err:
                        raw_errors.append(f"{spec.label}: {err}")
                    if v is not None:
                        observations_entered += 1
                    vals.append(v)

            raw_obs[spec.key_base] = vals

    st.subheader("Data check")
    st.write(f"**Observations entered:** {observations_entered}")

    if raw_errors:
        uniq = []
        for e in raw_errors:
            if e not in uniq:
                uniq.append(e)
        st.warning("Some entries looked invalid and will be treated as missing:\n\n" + "\n".join([f"• {e}" for e in uniq]))

    # Button FIRST
    calculate = st.button("Calculate risk", type="primary")

    # And the results container IMMEDIATELY BELOW it (so results appear below the form/button)
    manual_result_container = st.container()

    if calculate:
        try:
            feats: dict[str, Any] = {}

            for base_key, vals in raw_obs.items():
                any_measured = any(v is not None for v in vals)
                feats[f"{base_key}_was_measured"] = 1 if any_measured else 0

                mean_v, last_v = engineer_from_timepoints(vals)
                feats[f"{base_key}_mean"] = np.nan if mean_v is None else float(mean_v)
                feats[f"{base_key}_last"] = np.nan if last_v is None else float(last_v)

                if base_key.lower() == "mechvent":
                    if any_measured:
                        mv = [v for v in vals if v is not None]
                        feats["MechVent_prop_on"] = float(np.mean(mv)) if len(mv) else np.nan
                        feats["MechVent_last"] = float(mv[-1]) if len(mv) else np.nan
                    else:
                        feats["MechVent_prop_on"] = np.nan
                        feats["MechVent_last"] = np.nan

            X_imp, present_count, missing_cols, X_aligned = align_and_impute(feats, feature_columns, imputer)
            engineered_present_count = present_count

            if mode.startswith("Strict"):
                ok, reason = strict_gate_ok(engineered_present_count, observations_entered)
                if not ok:
                    with manual_result_container:
                        st.warning(
                            "Strict mode: insufficient data to provide a meaningful estimate.\n\n"
                            f"Reason: {reason}\n\n"
                            "Switch to **Estimate mode** if you still want an approximate result using typical values."
                        )
                    st.stop()

            prob = float(model.predict_proba(X_imp)[:, 1][0])

            # Persist result so it stays visible after reruns
            st.session_state["manual_result"] = dict(
                prob=prob,
                present_count=present_count,
                missing_cols=missing_cols,
                X_imp=X_imp,
                X_aligned=X_aligned,
                observations_entered=observations_entered,
                engineered_present_count=engineered_present_count,
            )

        except Exception as e:
            with manual_result_container:
                st.error("Something went wrong while predicting.")
                st.exception(e)

    # Always render the latest manual result (if present) BELOW the button/form
    res = st.session_state.get("manual_result")
    if res:
        render_result(
            container=manual_result_container,
            prob=res["prob"],
            present_count=res["present_count"],
            missing_cols=res["missing_cols"],
            X_imp=res["X_imp"],
            X_aligned=res["X_aligned"],
            observations_entered=res["observations_entered"],
            engineered_present_count=res["engineered_present_count"],
        )
