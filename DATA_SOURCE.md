# Dataset provenance

**Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, dataset ID 17. Official page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The project includes a CSV snapshot with 569 rows and 30 numeric predictors plus a binary diagnosis label. Features are computed from digitized fine-needle aspirate images and describe cell-nuclei measurements. In the included file, labels are normalized to `malignant` and `benign`; the modeling script maps malignant to the positive class (1).

The CSV was materialized from scikit-learn's public `load_breast_cancer(as_frame=True)` copy of the UCI data. Official scikit-learn reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Dataset attribution is retained here and in the report. Consult the official source for citation instructions and reuse terms. No patient-identifying information is included in this project file.
