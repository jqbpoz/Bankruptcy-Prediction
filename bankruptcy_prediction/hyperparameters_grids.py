# Hyperparameter grids for various models

LIGHTGBM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "num_leaves": [15, 31, 45],
    "subsample": [0.6, 0.8, 1.0]
}

RANDOM_FOREST_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

XGBOOST_GRID = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 10],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.6, 0.8, 1.0]
}

LOGISTIC_REGRESSION_GRID = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [0.1, 1, 10, 100],
    "solver": ["liblinear", "saga"]
}

# Mapping for easier access by model name
GRID_MAPPING = {
    "LightGBM": LIGHTGBM_GRID,
    "RandomForest": RANDOM_FOREST_GRID,
    "XGBoost": XGBOOST_GRID,
    "LogisticRegression": LOGISTIC_REGRESSION_GRID
}
