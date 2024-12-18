# Default hyperparameters for various models

DEFAULT_LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "metric": 'auc',
    "max_depth": 5,
    "num_leaves": 31,
    "subsample": 1.0
}

DEFAULT_RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "metric": "auc"
}

DEFAULT_XGBOOST_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "eval_metric": "auc"
}

DEFAULT_LOGISTIC_REGRESSION_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "metric": "auc"
}

# Mapping for easy access
DEFAULT_PARAMS_MAPPING = {
    "LightGBM": DEFAULT_LIGHTGBM_PARAMS,
    "RandomForest": DEFAULT_RANDOM_FOREST_PARAMS,
    "XGBoost": DEFAULT_XGBOOST_PARAMS,
    "LogisticRegression": DEFAULT_LOGISTIC_REGRESSION_PARAMS
}
