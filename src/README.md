# 📦 `src/` — Core Machine Learning Pipeline

This directory contains all **core source code** for the fraud detection project.
Each module is responsible for a **single, well-defined stage** of the machine learning workflow, following clean code and reproducibility principles.

---

## 📁 Directory Structure

```
src/
│── __init__.py
│── preprocessing.py
│── split.py
│── baseline_model.py
│── ensemble_model.py
│── training_pipeline.py
```

---

## 🧹 `preprocessing.py`

**Purpose:**
Handles dataset-specific preprocessing logic.

**Responsibilities:**

* Cleaning raw and processed datasets
* Encoding categorical features
* Scaling numerical features
* Ensuring data consistency before modeling

**Main functions:**

* `preprocess_fraud(df)`
* `preprocess_creditcard(df)`

These functions return **model-ready DataFrames**.

---

## ✂️ `split.py`

**Purpose:**
Create reproducible and stratified train–test splits.

**Responsibilities:**

* Preserve class distribution in imbalanced datasets
* Separate features and target variable

**Main function:**

* `stratified_split(df, target)`

---

## 📉 `baseline_model.py`

**Purpose:**
Define baseline models for performance comparison.

**Responsibilities:**

* Provide a simple, interpretable benchmark
* Establish a minimum acceptable performance level

**Main model:**

* Logistic Regression

**Main function:**

* `logistic_regression_model()`

---

## 🌲 `ensemble_model.py`

**Purpose:**
Define ensemble-based models for improved predictive performance.

**Responsibilities:**

* Train a Random Forest classifier
* Optionally apply hyperparameter tuning using cross-validation

**Main functions:**

* `random_forest_model(tuned=False)`

When `tuned=True`, the model uses `RandomizedSearchCV` for parameter optimization.

---

## 🧪 `training_pipeline.py`

**Purpose:**
Centralized training and evaluation logic.

**Responsibilities:**

* Train models on training data
* Generate predictions on test data
* Compute evaluation metrics consistently across models

**Main function:**

* `train_and_evaluate(X_train, y_train, X_test, y_test, model)`

Returns:

* Trained model
* Dictionary of evaluation metrics (e.g., precision, recall, F1-score, ROC-AUC)

---

## 🔁 Design Principles

* **Modular:** Each file handles one responsibility
* **Reusable:** Models and preprocessing can be reused across notebooks
* **Reproducible:** Fixed random states and stratified splits
* **Notebook-light:** Heavy logic lives in `src/`, not notebooks

---

## 📌 How `src/` is Used

All modules in this directory are imported and executed from:

* `notebooks/modeling.ipynp`
* Automated tests under `tests/`

The notebook acts as an **orchestrator**, while `src/` contains the implementation.

---

## ✅ Task-2 Alignment

This structure fully supports **Task-2: Modeling & Evaluation**, including:

* Baseline vs ensemble comparison
* Dataset-specific preprocessing
* Optional hyperparameter tuning
* Clean experiment separation

