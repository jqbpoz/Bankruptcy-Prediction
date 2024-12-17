import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
        self.kwargs = kwargs
        self.model = self._initialize_model(algorithm, **kwargs)

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

    def train(self, X_train, y_train):
        """Train the model with given data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return performance metrics."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return {"accuracy": accuracy, "report": report}

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
