import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, f1_score, roc_curve, \
    roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score, log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from bankruptcy_prediction.default_hyperparameters import DEFAULT_PARAMS_MAPPING
from tqdm import tqdm

class Model:
    def __init__(self, algorithm="RandomForest", **kwargs):
        """
        Initialize the Model with the selected algorithm.
        Supported algorithms:
            - RandomForest
            - LogisticRegression
            - XGBoost
            - LightGBM
        """
        self.algorithm = algorithm
        self.model = self._initialize_model(algorithm, **kwargs)
        self.evaluation_results = {}
        self.hyperparameters = kwargs


    def _initialize_model(self, algorithm, **kwargs):
        """Initialize the appropriate model based on the selected algorithm."""
        if algorithm == "RandomForest":
            return RandomForestClassifier(**kwargs)
        elif algorithm == "LogisticRegression":
            return LogisticRegression(**kwargs)
        elif algorithm == "XGBoost":
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **kwargs)
        elif algorithm == "LightGBM":
            return LGBMClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def fit(self, X_train, y_train ,**params):
        """Train the model with given data."""
        self.model.fit(X_train, y_train, **params)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return performance metrics."""
        y_pred = self.model.predict(X_test)
        confusion = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        self.evaluation_results = {"accuracy": accuracy, "report": report, "confusion": confusion}

    def predict(self, X):
        """Predict using the trained model."""
        return self.model.predict(X)

    def save(self, filepath):
        """Save the trained model to a file."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, algorithm="RandomForest", **kwargs):
        """Load a model from a file and reinitialize the Model class."""
        model = joblib.load(filepath)
        instance = cls(algorithm=algorithm, **kwargs)
        instance.model = model
        return instance


    def _format_results(self):
        """Format the evaluation results."""
        results = [
            f"Algorithm: {self.algorithm}",
            "Hyperparameters:"
        ]
        results.extend([f"  {param}: {value}" for param, value in self.hyperparameters.items()])
        results.append(f"Accuracy: {self.evaluation_results['accuracy']:.4f}")
        results.append("Classification Report:")
        results.append(self.evaluation_results['report'])
        results.append("Confusion Matrix:")
        results.append(str(self.evaluation_results['confusion']))
        return "\n".join(results)

    def print_results(self):
        """Print the evaluation results."""
        print(self._format_results())

    def save_results(self, filepath):
        """Save the evaluation results to a file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(self._format_results())
            print(f"Results saved to {filepath}")
        except (IOError, OSError) as e:
            print(f"Failed to save results to {filepath}: {e}")

    def tune_hyperparameters(self, X_train, y_train, param_grid=None, optimization_method="grid", scoring="f1", cv=5, n_iter=50):
        """
        Perform hyperparameter tuning on the training data.

        Args:
            X_train: Training features.
            y_train: Training labels.
            param_grid: Dictionary of hyperparameters to search. If None, use default from GRID_MAPPING.
            optimization_method: Optimization method - "grid" or "random".
            scoring: Metric to optimize. Default is "f1". Options include:
            - "accuracy": Percentage of correct classifications.
            - "precision": Precision.
            - "recall": Recall.
            - "f1": F1-score (harmonic mean of precision and recall).
            - "roc_auc": Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
            - "average_precision": Average precision.
            - "neg_log_loss": Negative logarithmic loss.
            - "neg_mean_squared_error": Negative mean squared error (MSE).
            - "neg_mean_absolute_error": Negative mean absolute error (MAE).
            - "neg_mean_absolute_percentage_error": Negative mean absolute percentage error (MAPE).
            cv: Number of cross-validation folds.
            n_iter: Number of iterations for random search (used if optimization_method="random").
        """
        if param_grid is None:
            from bankruptcy_prediction.hyperparameters_grids import GRID_MAPPING
            param_grid = GRID_MAPPING[self.algorithm]


        if optimization_method == "grid":
            search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                verbose=0
            )

        elif optimization_method == "random":
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                scoring=scoring,
                cv=cv,
                n_iter=n_iter,
                verbose=0,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported optimization method: {optimization_method}")


        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.hyperparameters = search.best_params_
        return self.model, self.hyperparameters

    def plot_roc_curve(self, X_test, y_test):
        """Plot the ROC curve for the model."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_pr_curve(self, X_test, y_test):
        """Plot the Precision-Recall curve for the model."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.fill_between(recall, precision, color='darkorange', alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()