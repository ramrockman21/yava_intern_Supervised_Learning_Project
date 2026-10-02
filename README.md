# Week 4 — Supervised Learning Model Implementation

A reproducible binary-classification project using the public Breast Cancer Wisconsin (Diagnostic) dataset. The supplied CSV means the training script does **not** need network access.

## Project objective

Train, validate, and evaluate a supervised model that classifies a breast-mass sample as **malignant** or **benign** from 30 numeric features computed from a digitized fine-needle aspirate (FNA) image. Malignant is the positive class. This is an educational benchmark only—not a medical device or diagnostic recommendation.

## Folder guide

- `data/breast_cancer_wisconsin_diagnostic.csv` — dataset snapshot used by the script (569 rows, 30 feature columns, one `diagnosis` target column).
- `src/train_model.py` — end-to-end data checks, stratified holdout, 5-fold CV model comparison, final evaluation, and artifact export.
- `outputs/` — example run outputs: validation table, holdout metrics, classification report, confusion matrix, predictions, run metadata, and fitted pipeline.
- `report/Week4_Supervised_Learning_Report.docx` — detailed project report.

## Run in VS Code

1. Install Python 3.10 or newer and open this project folder in VS Code.
2. In the integrated terminal, create and activate a virtual environment:

   **Windows (PowerShell)**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies and run:
   ```bash
   python -m pip install -r requirements.txt
   python src/train_model.py
   ```
4. Review the new files in `outputs/`. Re-running overwrites the prior outputs using the same random seed and split configuration.

## Method summary

- Target mapping: `malignant` → 1; `benign` → 0.
- Split: stratified 80% train / 20% held-out test, random seed 42.
- Validation: shuffled 5-fold stratified cross-validation on the training set only.
- Candidate models: logistic regression, RBF support vector machine, random forest, and a majority-class dummy baseline.
- Selection: highest mean CV ROC-AUC among the three trained candidates. Test data is reserved for the one final evaluation.
- Preprocessing: StandardScaler is inside the linear/SVM model pipelines so it is fitted within each CV training fold. Random Forest uses raw numeric features.
- Positive-class interpretation: recall is sensitivity for malignant samples; a false negative is an actual malignant case predicted benign.

## Dataset attribution

Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>. The included CSV is a local snapshot of the same dataset exposed by scikit-learn's `load_breast_cancer` helper. Review the source page for dataset citation and reuse terms.

## Reproducibility and limitations

The seed, split, class mapping, model-selection rule, and scikit-learn version are recorded in `outputs/run_metadata.json`. Exact results may vary slightly across library versions/platforms. This small historical dataset is not representative evidence of real-world clinical performance; external validation, calibration, threshold selection based on clinical costs, and expert review would be required before any real-world use.
