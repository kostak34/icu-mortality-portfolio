# ICU Mortality Prediction (Portfolio Demo)

This repository contains a supervised ML project that predicts ICU mortality from the first 48h of irregularly-sampled clinical measurements.

**Status:** Portfolio / learning artifact (not for clinical deployment).

## What’s included
- Data parsing from per-patient text files
- Feature engineering (mean / last / was_measured + special handling)
- Model training (Random Forest) + calibration
- Threshold selection for a safety-oriented operating point
- Evaluation (ROC/PR, confusion matrix, cross-validation)
- Interpretability (SHAP local explanations)

## Quickstart
1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2) Install requirements
```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

3) Place data locally (do NOT commit data)
```
data/
  raw/
    outcomes_a.txt
    set-a/
      132539.txt
      ...
```

4) Run the pipeline (adjust filenames to match your project)
```bash
python scripts/step_01_load_raw.py
python scripts/step_02_batch_features.py
python scripts/step_03_explore_and_clean.py
python scripts/step_04_train_logreg.py
python scripts/step_05_train_rf_calibrate.py
```

5) Run inference (predict on a single patient file)
```bash
python predict_patient_file.py --patient_file data/raw/set-a/132539.txt --threshold 0.223
```

## Outputs
- Saved model artifact: `outputs/model_bundle.joblib` (created after adding the save snippet)
- Figures: `docs/figures/` (ROC, PR, calibration, confusion matrix, SHAP screenshots)
- Model Card PDF: `docs/ICU_Mortality_Model_Card_v1.pdf`

## Notes
- This repo intentionally excludes patient-level data (`data/` is gitignored).
- Add clear disclaimers if you deploy a demo (Streamlit/Gradio): “Educational demo only.”
