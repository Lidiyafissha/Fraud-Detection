# ==============================
# TESTING & ROBUSTNESS NOTE
# ==============================
# To improve reliability, basic validation and error handling should be
# added to the preprocessing and training steps.
#
# This includes:
# - Asserting that required columns exist before processing
# - Verifying target column presence and type
# - Handling missing values or invalid data types gracefully
# - Failing fast with clear error messages instead of silent crashes
#
# These checks can be tested independently while preserving the current
# modular structure (preprocessing, splitting, modeling, training).

import pandas as pd
import pytest

from src.preprocessing import preprocess_fraud, preprocess_creditcard
from src.training_pipeline import train_and_evaluate
from src.baseline_model import logistic_regression_model


# =====================================================
# PREPROCESSING VALIDATION TESTS
# =====================================================

def test_preprocess_fraud_requires_columns():
    """
    preprocess_fraud should fail fast if required columns are missing.
    """
    df = pd.DataFrame({
        "age": [25, 30],
        "purchase_value": [100, 200]
    })

    with pytest.raises((AssertionError, KeyError, ValueError)):
        preprocess_fraud(df)


def test_preprocess_creditcard_requires_columns():
    """
    preprocess_creditcard should fail if critical CC columns are missing.
    """
    df = pd.DataFrame({
        "Amount": [10.5, 20.3]
    })

    with pytest.raises((AssertionError, KeyError, ValueError)):
        preprocess_creditcard(df)


# =====================================================
# TRAINING PIPELINE VALIDATION TESTS
# =====================================================

def test_training_fails_without_target():
    """
    Training must not run if y_train is None.
    """
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y = None

    model = logistic_regression_model()

    with pytest.raises((AssertionError, ValueError, TypeError)):
        train_and_evaluate(X, y, X, y, model)


def test_training_fails_on_length_mismatch():
    """
    X and y length mismatch should raise a clear error.
    """
    X = pd.DataFrame({"f1": [1, 2, 3]})
    y = pd.Series([0, 1])

    model = logistic_regression_model()

    with pytest.raises((AssertionError, ValueError)):
        train_and_evaluate(X, y, X, y, model)

