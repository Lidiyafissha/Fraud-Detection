from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_
