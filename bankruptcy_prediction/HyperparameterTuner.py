from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class HyperparameterTuner:
    def __init__(self, model_class, param_grid, optimization_method="grid", scoring="f1", cv=5, n_iter=50):
        """
        Hyperparameter tuner to optimize model parameters.

        Args:
            model_class: The model class (e.g., RandomForestClassifier).
            param_grid: Dictionary of hyperparameters to search.
            optimization_method: Optimization method - "grid" or "random".
            scoring: Metric to optimize. Default is "f1".
            cv: Number of cross-validation folds.
            n_iter: Number of iterations for random search (used if optimization_method="random").
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.optimization_method = optimization_method
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter

    def tune(self, X_train, y_train):
        """
        Perform hyperparameter tuning on the training data.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            best_model: Trained model with the best parameters.
            best_params: Best hyperparameters found.
        """
        if self.optimization_method == "grid":
            search = GridSearchCV(
                estimator=self.model_class(),
                param_grid=self.param_grid,
                scoring=make_scorer(f1_score, average="binary"),
                cv=self.cv,
                verbose=1
            )
        elif self.optimization_method == "random":
            search = RandomizedSearchCV(
                estimator=self.model_class(),
                param_distributions=self.param_grid,
                scoring=make_scorer(f1_score, average="binary"),
                cv=self.cv,
                n_iter=self.n_iter,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported optimization method: {self.optimization_method}")

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        return best_model, best_params
