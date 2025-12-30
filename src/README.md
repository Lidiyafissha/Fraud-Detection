# 📦 `src/` — Core Machine Learning Pipeline

This directory contains all **core source code** for the fraud detection project. Each module is responsible for a **single, well-defined stage** of the machine learning workflow, following clean code and reproducibility principles.

---

## 📁 Directory Structure

```
src/
│── __init__.py
│── preprocessing.py          # Dataset-specific preprocessing
│── split.py                  # Stratified train-test splitting
│── imbalance.py              # SMOTE for handling class imbalance
│── baseline_model.py         # Logistic Regression baseline
│── ensemble_model.py         # Random Forest ensemble
│── hyperparameter_tuning.py  # Cross-validation hyperparameter tuning
│── modeltuning.py           # Simple hyperparameter tuning utilities
│── evaluation.py             # Model evaluation metrics
│── training_pipeline.py      # Unified training & evaluation pipeline
```

---

## 🧹 `preprocessing.py`

**Purpose:** Handles dataset-specific preprocessing logic for both fraud and credit card datasets.

**Responsibilities:**
- Data validation and cleaning
- Handling missing values
- Encoding categorical features
- Scaling numerical features
- Ensuring data consistency before modeling

**Main Functions:**

- `preprocess_fraud(df: pd.DataFrame) -> pd.DataFrame`
  - Preprocesses e-commerce fraud dataset
  - Validates required columns
  - Handles missing values (median for numeric, "UNKNOWN" for categorical)
  - Validates binary target variable

- `preprocess_creditcard(df: pd.DataFrame) -> pd.DataFrame`
  - Preprocesses credit card transaction dataset
  - Handles PCA-transformed features (V1-V28)
  - Validates target column

**Returns:** Model-ready DataFrames with consistent structure.

---

## ✂️ `split.py`

**Purpose:** Create reproducible and stratified train–test splits.

**Responsibilities:**
- Preserve class distribution in imbalanced datasets
- Separate features and target variable
- Ensure reproducibility with fixed random states

**Main Function:**

- `stratified_split(df, target, test_size=0.2, random_state=42)`
  - Performs stratified train-test split
  - Maintains class distribution across splits
  - Returns: `X_train, X_test, y_train, y_test`

**Key Features:**
- Uses `train_test_split` with `stratify` parameter
- Default 80/20 split ratio
- Fixed random state for reproducibility

---

## ⚖️ `imbalance.py`

**Purpose:** Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

**Responsibilities:**
- Apply SMOTE to training data only (never on test data)
- Balance class distribution for better model training
- Generate synthetic fraud cases

**Main Function:**

- `apply_smote(X, y, random_state=42)`
  - Applies SMOTE oversampling to training data
  - Returns resampled `X_res, y_res`
  - Uses fixed random state for reproducibility

**Usage:**
```python
from src.imbalance import apply_smote

X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
```

---

## 📉 `baseline_model.py`

**Purpose:** Define baseline models for performance comparison.

**Responsibilities:**
- Provide a simple, interpretable benchmark
- Establish a minimum acceptable performance level
- Serve as a reference point for ensemble models

**Main Function:**

- `logistic_regression_model()`
  - Returns configured Logistic Regression model
  - Uses class weighting to handle imbalance
  - Includes regularization (L2 penalty)

**Configuration:**
- `class_weight='balanced'` - Handles class imbalance
- `max_iter=1000` - Sufficient iterations for convergence
- `random_state=42` - Reproducibility

**Use Case:** Baseline comparison against Random Forest ensemble.

---

## 🌲 `ensemble_model.py`

**Purpose:** Define ensemble-based models for improved predictive performance.

**Responsibilities:**
- Train Random Forest classifier
- Optionally apply hyperparameter tuning
- Capture non-linear relationships and complex patterns

**Main Function:**

- `random_forest_model(tuned=False)`
  - Returns Random Forest classifier
  - When `tuned=True`, uses optimized hyperparameters
  - Default parameters optimized for fraud detection

**Configuration:**
- `n_estimators=100` - Number of trees
- `max_depth=10` - Tree depth limit
- `min_samples_split=5` - Minimum samples to split
- `class_weight='balanced'` - Handles imbalance
- `random_state=42` - Reproducibility

**Use Case:** Primary production model for fraud detection.

---

## 🎯 `hyperparameter_tuning.py`

**Purpose:** Advanced hyperparameter tuning using cross-validation with imbalanced-aware metrics.

**Responsibilities:**
- Tune Logistic Regression hyperparameters
- Tune Random Forest hyperparameters
- Use PR-AUC and F1-score as optimization metrics
- Provide cross-validation performance evaluation

**Main Functions:**

- `tune_logistic_regression(X_train, y_train, cv=5, scoring='f1', random_state=42)`
  - Tunes Logistic Regression using RandomizedSearchCV
  - Searches over C, penalty, solver, class_weight
  - Returns best model and tuning results

- `tune_random_forest(X_train, y_train, cv=5, scoring='f1', random_state=42, n_iter=50)`
  - Tunes Random Forest using RandomizedSearchCV
  - Searches over n_estimators, max_depth, min_samples_split, min_samples_leaf
  - Returns best model and tuning results

- `tune_model(model_type, X_train, y_train, cv=5, scoring='f1', random_state=42)`
  - Unified interface for tuning any model type
  - Supports 'logistic' and 'random_forest'

- `evaluate_cv_performance(model, X_train, y_train, cv=5, scoring='f1')`
  - Evaluates model using cross-validation
  - Returns mean and std of CV scores

- `get_cv_strategy(n_splits=5, shuffle=True, random_state=42)`
  - Returns StratifiedKFold for cross-validation
  - Ensures class distribution in each fold

**Custom Scorers:**
- `_pr_auc_scorer()` - Precision-Recall AUC scorer
- `_f1_scorer()` - F1-score scorer

**Key Features:**
- Uses `StratifiedKFold` for imbalanced datasets
- Multiple scoring metrics (F1, PR-AUC)
- Randomized search for efficiency
- Comprehensive parameter grids

---

## 🔧 `modeltuning.py`

**Purpose:** Simple hyperparameter tuning utilities.

**Responsibilities:**
- Provide basic hyperparameter tuning functionality
- Grid search implementation
- Model optimization utilities

**Main Function:**

- `tune_hyperparameters(model, param_grid, X_train, y_train)`
  - Performs grid search hyperparameter tuning
  - Returns best model and best parameters

**Use Case:** Lightweight tuning for quick experiments.

---

## 📊 `evaluation.py`

**Purpose:** Model evaluation using imbalanced-aware metrics.

**Responsibilities:**
- Compute evaluation metrics consistently
- Focus on metrics suitable for imbalanced datasets
- Provide standardized evaluation interface

**Main Function:**

- `evaluate(model, X_test, y_test)`
  - Evaluates trained model on test data
  - Returns dictionary with metrics:
    - `F1`: F1-score
    - `AUC_PR`: Average Precision Score (PR-AUC)
    - `ConfusionMatrix`: Confusion matrix array

**Metrics Used:**
- **F1-Score**: Balance between precision and recall
- **PR-AUC**: Precision-Recall Area Under Curve (preferred over ROC-AUC for imbalanced data)
- **Confusion Matrix**: Detailed error analysis

---

## 🧪 `training_pipeline.py`

**Purpose:** Centralized training and evaluation logic.

**Responsibilities:**
- Train models on training data
- Generate predictions on test data
- Compute evaluation metrics consistently across models
- Provide unified interface for model training

**Main Function:**

- `train_and_evaluate(X_train, y_train, X_test, y_test, model)`
  - Trains model on training data
  - Evaluates on test data
  - Returns:
    - Trained model
    - Dictionary of evaluation metrics:
      - `F1`: F1-score
      - `PR_AUC`: Precision-Recall AUC
      - `Confusion_Matrix`: Confusion matrix

**Usage:**
```python
from src.training_pipeline import train_and_evaluate
from src.baseline_model import logistic_regression_model

model = logistic_regression_model()
trained_model, metrics = train_and_evaluate(
    X_train, y_train, X_test, y_test, model
)
print(metrics)
```

---

## 🔁 Design Principles

- **Modular:** Each file handles one responsibility (Single Responsibility Principle)
- **Reusable:** Models and preprocessing can be reused across notebooks and scripts
- **Reproducible:** Fixed random states and stratified splits ensure consistent results
- **Notebook-light:** Heavy logic lives in `src/`, notebooks act as orchestrators
- **Testable:** All functions are unit-testable with clear inputs/outputs
- **Documented:** Clear function signatures and docstrings

---

## 📌 Usage Workflow

### Basic Training Pipeline

```python
import sys
import os
sys.path.append(os.path.abspath(".."))

from src.preprocessing import preprocess_fraud
from src.split import stratified_split
from src.baseline_model import logistic_regression_model
from src.ensemble_model import random_forest_model
from src.training_pipeline import train_and_evaluate

# 1. Load and preprocess data
df = pd.read_csv('../data/processed/fraud_final.csv')
df = preprocess_fraud(df)

# 2. Split data
X_train, X_test, y_train, y_test = stratified_split(df, target="class")

# 3. Train baseline model
lr = logistic_regression_model()
lr_model, lr_metrics = train_and_evaluate(X_train, y_train, X_test, y_test, lr)

# 4. Train ensemble model
rf = random_forest_model()
rf_model, rf_metrics = train_and_evaluate(X_train, y_train, X_test, y_test, rf)

# 5. Compare results
print("Logistic Regression:", lr_metrics)
print("Random Forest:", rf_metrics)
```

### With SMOTE

```python
from src.imbalance import apply_smote

# Apply SMOTE to training data only
X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

# Train on resampled data
model, metrics = train_and_evaluate(
    X_train_resampled, y_train_resampled, X_test, y_test, rf
)
```

### With Hyperparameter Tuning

```python
from src.hyperparameter_tuning import tune_random_forest

# Tune Random Forest
best_rf, tuning_results = tune_random_forest(
    X_train, y_train, cv=5, scoring='f1', random_state=42
)

# Evaluate tuned model
tuned_model, tuned_metrics = train_and_evaluate(
    X_train, y_train, X_test, y_test, best_rf
)
```

---

## 📌 Where `src/` is Used

All modules in this directory are imported and executed from:

- **`notebooks/modeling.ipynb`** - Main modeling notebook
- **`notebooks/shap-explainability.ipynb`** - SHAP analysis notebook
- **`scripts/train_models.py`** - Automated training scripts
- **`tests/test_*.py`** - Unit tests for each module

The notebooks and scripts act as **orchestrators**, while `src/` contains the **implementation**.

---

## ✅ Task Alignment

This structure fully supports:

- **Task 2: Model Building, Training & Evaluation**
  - Baseline vs ensemble comparison
  - Dataset-specific preprocessing
  - Hyperparameter tuning
  - Clean experiment separation

- **Task 3: SHAP Explainability**
  - Model loading and prediction
  - Feature importance analysis
  - Model explainability integration

---

## 🧪 Testing

All modules have corresponding unit tests in `tests/`:

- `test_training_pipeline.py` - Tests training pipeline
- `test_hyperparameter_tuning.py` - Tests hyperparameter tuning
- `test_shap_pipeline.py` - Tests SHAP integration

Run tests with:
```bash
pytest tests/
```

---

## 📝 Notes

- All random states are fixed to `42` for reproducibility
- Models are designed for binary classification (fraud detection)
- Metrics focus on imbalanced learning (F1, PR-AUC) rather than accuracy
- Preprocessing functions validate input data and raise clear errors
- All functions return consistent data structures for easy integration
