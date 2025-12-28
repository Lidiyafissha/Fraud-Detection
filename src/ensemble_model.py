from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def random_forest_model(tuned=False):
    if not tuned:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1   # OK: no CV here
        )

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=1      # 🔴 must be 1
    )

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    return RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=6,     # keep small
        cv=3,
        scoring="average_precision",
        n_jobs=1,     # 🔴 CRITICAL FIX
        verbose=1,
        random_state=42
    )
