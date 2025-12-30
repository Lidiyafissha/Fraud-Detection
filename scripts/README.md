# 📜 `scripts/` — Automation & Utility Scripts

This directory contains **standalone scripts** for automating common tasks in the fraud detection pipeline. These scripts can be run from the command line and are designed for production use, CI/CD integration, and automated workflows.

---

## 📁 Directory Structure

```
scripts/
│── __init__.py
│── train_models.py      # Automated model training script
│── shap_analysis.py      # SHAP explainability analysis utility
```

---

## 🚀 `train_models.py`

**Purpose:** Automated script for training fraud detection models from the command line.

**Description:**
This script provides a command-line interface for training fraud detection models with optional hyperparameter tuning. It handles the complete training pipeline from data loading to model saving.

### Features

- **Multiple Model Types:** Support for Random Forest and Logistic Regression
- **Hyperparameter Tuning:** Optional tuning with RandomizedSearchCV or GridSearchCV
- **Automatic Model Saving:** Saves trained models and feature lists
- **Performance Reporting:** Displays evaluation metrics after training
- **Reproducible:** Uses fixed random states for consistent results

### Usage

#### Basic Training (Random Forest - Default)

```bash
python scripts/train_models.py
```

#### Train Logistic Regression

```bash
python scripts/train_models.py --model logistic_regression
```

#### Train with Hyperparameter Tuning

```bash
python scripts/train_models.py --tune
```

#### Train with Custom Search Method

```bash
# Randomized search (default, faster)
python scripts/train_models.py --tune --search random --n-iter 50

# Grid search (exhaustive, slower)
python scripts/train_models.py --tune --search grid
```

#### Complete Example

```bash
python scripts/train_models.py \
    --model random_forest \
    --tune \
    --search random \
    --n-iter 100
```

### Command-Line Arguments

| Argument   | Type | Default         | Description                                          |
| ---------- | ---- | --------------- | ---------------------------------------------------- |
| `--tune`   | flag | `False`         | Enable hyperparameter tuning                         |
| `--model`  | str  | `random_forest` | Model type: `random_forest` or `logistic_regression` |
| `--search` | str  | `random`        | Search method: `random` or `grid`                    |
| `--n-iter` | int  | `50`            | Number of iterations for randomized search           |

### Output

The script saves the following files to the `models/` directory:

- **`fraud_rf.pkl`** or **`fraud_lr.pkl`**: Trained model (joblib format)
- **`fraud_features.pkl`**: List of feature names used for inference

### Example Output

```
Loading and preprocessing data...
Splitting data...
Training random_forest model...

✅ random_forest model trained and saved to models/fraud_rf.pkl
📊 Test Metrics:
   F1 Score: 0.6404
   PR-AUC: 0.7226
   Confusion Matrix:
[[22202  1174]
 [  745  1709]]
```

### Integration

This script is used by:

- **CI/CD pipelines** for automated model retraining
- **Production workflows** for model deployment
- **Experimentation** for quick model training

---

## 🔍 `shap_analysis.py`

**Purpose:** Utility script for loading SHAP components needed for model explainability analysis.

**Description:**
This script provides a convenient function to load all components required for SHAP explainability analysis, including the trained model, test data, and SHAP explainer. It's designed to be imported by notebooks and other analysis scripts.

### Features

- **Model Loading:** Loads trained Random Forest model
- **Data Preparation:** Preprocesses and splits data consistently
- **Feature Alignment:** Ensures test data matches model features
- **Memory Optimization:** Samples test data to avoid memory issues
- **SHAP Initialization:** Creates TreeExplainer and computes SHAP values

### Usage

#### In Python Scripts/Notebooks

```python
import sys
import os
sys.path.append(os.path.abspath(".."))

from scripts.shap_analysis import load_shap_components

# Load all SHAP components
model, explainer, shap_values, X_test, y_test = load_shap_components()

# Now you can use SHAP for analysis
import shap
shap.summary_plot(shap_values, X_test)
```

#### In Jupyter Notebooks

```python
# notebooks/shap-explainability.ipynb
from scripts.shap_analysis import load_shap_components

model, explainer, shap_values, X_test, y_test = load_shap_components()
```

### Function Signature

```python
def load_shap_components():
    """
    Load all components needed for SHAP explainability analysis.

    Returns:
        tuple: (model, explainer, shap_values, X_test, y_test)
            - model: Trained Random Forest model
            - explainer: SHAP TreeExplainer instance
            - shap_values: SHAP values for test data
            - X_test: Test features (sampled to 1000 instances)
            - y_test: Test labels (corresponding to sampled X_test)
    """
```

### Return Values

1. **`model`**: Trained Random Forest classifier (joblib-loaded)
2. **`explainer`**: SHAP TreeExplainer instance for the model
3. **`shap_values`**: SHAP values array for test instances
4. **`X_test`**: Test feature DataFrame (sampled to 1000 rows)
5. **`y_test`**: Test labels Series (corresponding to sampled X_test)

### Data Processing

The function performs the following steps:

1. **Load Model:** Loads `models/fraud_rf.pkl` and `models/fraud_features.pkl`
2. **Load Data:** Reads `data/processed/fraud_final.csv`
3. **Preprocess:** Applies `preprocess_fraud()` to clean data
4. **Split:** Uses `stratified_split()` for consistent train/test split
5. **Align Features:** Ensures test data columns match model features
6. **Sample:** Samples 1000 test instances to avoid memory issues
7. **Compute SHAP:** Initializes TreeExplainer and computes SHAP values

### Memory Considerations

- Test data is sampled to **1000 instances** to reduce memory usage
- SHAP values are computed only for the sampled subset
- Original test set indices are preserved for reference

### Integration

This script is used by:

- **`notebooks/shap-explainability.ipynb`** - Task-3 SHAP analysis
- **SHAP visualization scripts** - For generating explainability reports
- **Model interpretation tools** - For understanding model decisions

### Example Usage in Notebook

```python
# Cell 1: Load components
from scripts.shap_analysis import load_shap_components

model, explainer, shap_values, X_test, y_test = load_shap_components()

# Cell 2: Global feature importance
import shap
shap.summary_plot(shap_values, X_test)

# Cell 3: Individual predictions
y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
tp_idx = X_test[(y_test == 1) & (y_pred == 1)].index[0]
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])
```

---

## 🔁 Design Principles

- **Standalone:** Scripts can be run independently from command line
- **Reusable:** Functions can be imported by notebooks and other scripts
- **Reproducible:** Fixed random states ensure consistent results
- **Well-Documented:** Clear docstrings and usage examples
- **Error Handling:** Validates inputs and provides clear error messages

---

## 📌 Dependencies

Both scripts require:

- **Project structure:** Must be run from project root or with proper path setup
- **Trained models:** `train_models.py` creates models, `shap_analysis.py` loads them
- **Data files:** Both scripts expect processed data in `data/processed/`
- **Source modules:** Both import from `src/` directory

### Required Files

For `train_models.py`:

- `data/processed/fraud_final.csv` - Processed fraud dataset

For `shap_analysis.py`:

- `models/fraud_rf.pkl` - Trained Random Forest model
- `models/fraud_features.pkl` - Feature list
- `data/processed/fraud_final.csv` - Processed fraud dataset

---

## 🧪 Testing

Scripts are tested indirectly through:

- **Integration tests** in `tests/test_shap_pipeline.py`
- **Notebook execution** in `notebooks/shap-explainability.ipynb`
- **Manual testing** via command-line execution

---

## 📝 Notes

- **Path Setup:** Scripts use `sys.path` manipulation to import from `src/`
- **Relative Paths:** Scripts use relative paths assuming execution from project root
- **Model Format:** Models are saved using `joblib` for efficient serialization
- **SHAP Sampling:** Test data is sampled to 1000 instances for memory efficiency
- **Feature Alignment:** `shap_analysis.py` ensures test features match model expectations

---

## 🚀 Quick Start

### Train a Model

```bash
# Train Random Forest (default)
python scripts/train_models.py

# Train with hyperparameter tuning
python scripts/train_models.py --tune
```

### Analyze Model with SHAP

```python
# In a notebook or Python script
from scripts.shap_analysis import load_shap_components
import shap

model, explainer, shap_values, X_test, y_test = load_shap_components()
shap.summary_plot(shap_values, X_test)
```

---

## ✅ Task Alignment

These scripts support:

- **Task 2: Model Building, Training & Evaluation**

  - `train_models.py` automates model training pipeline
  - Supports both baseline and ensemble models
  - Includes hyperparameter tuning capabilities

- **Task 3: SHAP Explainability**
  - `shap_analysis.py` provides SHAP components for analysis
  - Enables model explainability and feature importance analysis
  - Supports both global and local explanations
