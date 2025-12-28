# ==============================
# BASELINE MODEL EVALUATION
# ==============================
# We establish a clear baseline using Logistic Regression, a simple and
# interpretable linear model commonly used in fraud detection.
#
# The baseline is trained on the same preprocessed features and evaluated
# using a stratified train–test split to preserve class imbalance.
#
# Performance metrics (Precision, Recall, F1-score, ROC-AUC / PR-AUC)
# are computed on the test set and later compared against ensemble models
# to quantify the added value of more complex approaches.
#
# This ensures that any performance improvement is meaningful and not
# a result of data leakage or evaluation inconsistency.

from sklearn.linear_model import LogisticRegression

def logistic_regression_model():
    """
    Interpretable baseline model.
    """
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
