"""
Model Training Pipeline for Addis Ababa Housing Price Predictor
"""

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data_generator import generate_housing_data
from .utils import (
    ensure_dir,
    get_timestamp,
    regression_metrics,
    save_json,
    save_model,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {},
        "description": "Simple linear regression baseline",
    },
    "Ridge": {
        "model": Ridge(random_state=42),
        "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "description": "Ridge regression with L2 regularization",
    },
    "Lasso": {
        "model": Lasso(random_state=42),
        "params": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        "description": "Lasso regression with L1 regularization",
    },
    "ElasticNet": {
        "model": ElasticNet(random_state=42),
        "params": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.7, 0.9]},
        "description": "ElasticNet with both L1 and L2 regularization",
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "description": "Decision tree regressor",
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        "description": "Random forest ensemble",
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
        },
        "description": "Gradient boosting ensemble",
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 1.0],
        },
        "description": "AdaBoost ensemble",
    },
}

# ============================================
# DATA PROCESSING
# ============================================


class DataPreprocessor:
    """Professional data preprocessing pipeline."""

    def __init__(self) -> None:
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "price"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit preprocessor and transform data."""
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Encode categorical variables
        X_encoded = self._encode_categorical(X)

        # Scale numeric features
        X_scaled = self._scale_features(X_encoded)

        # Store feature names
        self.feature_names = X_scaled.columns.tolist()

        return X_scaled, y

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        X = df.drop(target_col, axis=1) if target_col else df.copy()

        # Encode categorical
        X_encoded = self._encode_categorical(X, fit=False)

        # Scale features
        X_scaled = self._scale_features(X_encoded, fit=False)

        return X_scaled

    def _encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns with LabelEncoder."""
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                X_encoded[col] = self.encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                # Handle unseen categories
                if col not in self.encoders:
                    # If encoder missing, fallback to fit behavior for safety
                    self.encoders[col] = LabelEncoder()
                    X_encoded[col] = self.encoders[col].fit_transform(X_encoded[col].astype(str))
                else:
                    known_values = set(self.encoders[col].classes_)
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: x if x in known_values else "unknown"
                    )
                    X_encoded[col] = self.encoders[col].transform(X_encoded[col].astype(str))

        return X_encoded

    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features."""
        if fit:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            assert self.scaler is not None, "Scaler has not been fitted."
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return X_scaled

    def save(self, filepath: str) -> None:
        """Save preprocessor to disk."""
        ensure_dir(os.path.dirname(filepath))
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "encoders": self.encoders,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                },
                f,
            )
        print("✅ Preprocessor saved to: {}".format(filepath))

    def load(self, filepath: str) -> None:
        """Load preprocessor from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.encoders = data["encoders"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]


# ============================================
# MODEL TRAINING
# ============================================


class ModelTrainer:
    """Professional model training pipeline."""

    def __init__(
        self,
        models_config: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        n_iter: int = 20,
    ) -> None:
        self.models_config: Dict[str, Any] = models_config or MODEL_CONFIG
        self.cv_folds: int = cv_folds
        self.n_iter: int = n_iter
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.results: Dict[str, Dict[str, Any]] = {}
        self.trained_models: Dict[str, Any] = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_importance: Optional[List[str]] = None

    def train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> None:
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print("\n📊 Data Split:")
        # At this point X_train/X_test are set, so it's safe to use len()
        assert self.X_train is not None and self.X_test is not None
        print("  - Training: {:,} samples".format(len(self.X_train)))
        print("  - Testing:  {:,} samples".format(len(self.X_test)))

    def train_all_models(self, use_hyperparameter_tuning: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train all models with optional hyperparameter tuning."""
        # Ensure training data is available for mypy and runtime
        assert self.X_train is not None and self.y_train is not None, "Call train_test_split first."
        print("\n" + "=" * 60)
        print("🚀 TRAINING MODELS")
        print("=" * 60)

        for name, config in self.models_config.items():
            print("\n📊 Training {}...".format(name))
            print("   {}".format(config.get("description", "")))

            model = config["model"]
            params = config.get("params", {})

            if use_hyperparameter_tuning and params:
                # Hyperparameter tuning
                best_model = self._hyperparameter_tune(model, params, name)
            else:
                # Train without tuning
                best_model = model.fit(self.X_train, self.y_train)  # type: ignore[arg-type]

            # Evaluate model
            y_pred = best_model.predict(self.X_test)  # type: ignore[arg-type]
            metrics = regression_metrics(self.y_test, y_pred)  # type: ignore[arg-type]

            # Add cross-validation score
            cv_scores = cross_val_score(
                best_model, self.X_train, self.y_train, cv=self.cv_folds, scoring="r2"
            )
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()

            self.results[name] = metrics
            self.trained_models[name] = best_model

            print(
                "   R²: {r2:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})".format(
                    r2=metrics["r2"],
                    cv_mean=metrics["cv_mean"],
                    cv_std=metrics["cv_std"],
                )
            )
            print("   RMSE: {:,.2f} ETB".format(metrics["rmse"]))
            print("   MAE: {:,.2f} ETB".format(metrics["mae"]))

        # Find best model
        if not self.results:
            raise RuntimeError("No models were trained successfully.")
        self.best_model_name = max(self.results, key=lambda x: self.results[x]["r2"])
        self.best_model = self.trained_models[self.best_model_name]

        print("\n" + "=" * 60)
        print("BEST MODEL: {}".format(self.best_model_name))
        print("   R² Score: {:.4f}".format(self.results[self.best_model_name]["r2"]))
        print("=" * 60)

        return self.results

    def _hyperparameter_tune(self, model: Any, params: Dict[str, Any], name: str) -> Any:
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        print("   Tuning hyperparameters...")

        # Reduce search space for large parameter grids
        if len(params) > 3:
            search = RandomizedSearchCV(
                model,
                params,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring="r2",
                random_state=42,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(model, params, cv=self.cv_folds, scoring="r2", n_jobs=-1)

        search.fit(self.X_train, self.y_train)  # type: ignore[arg-type]

        print("   Best params: {}".format(search.best_params_))

        return search.best_estimator_

    def create_ensemble(self) -> None:
        """Create a voting ensemble from trained models."""
        print("CREATING ENSEMBLE MODEL")
        print("=" * 60)

        if not self.results:
            print("No trained models available to create an ensemble.")
            return

        # Use top 3 models for ensemble
        top_models = sorted(self.results.items(), key=lambda x: x[1]["r2"], reverse=True)[:3]

        estimators: List[Tuple[str, Any]] = []
        for name, _ in top_models:
            estimators.append((name, self.trained_models[name]))

        ensemble = VotingRegressor(estimators)
        # Ensure training data present
        assert self.X_train is not None and self.y_train is not None
        ensemble.fit(self.X_train, self.y_train)  # type: ignore[arg-type]

        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test)  # type: ignore[arg-type]
        metrics = regression_metrics(self.y_test, y_pred)  # type: ignore[arg-type]

        self.results["Ensemble (Top 3)"] = metrics
        self.trained_models["Ensemble (Top 3)"] = ensemble

        print("Ensemble created from: {}".format(", ".join([e[0] for e in estimators])))
        print("   R²: {:.4f}".format(metrics["r2"]))
        print("   RMSE: {:,.2f} ETB".format(metrics["rmse"]))

    def get_feature_importance(self, model: Any) -> Optional[pd.DataFrame]:
        """Extract feature importance from tree-based models."""
        # Ensure training features exist
        if self.X_train is None:
            return None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        else:
            return None

        features = self.feature_importance or list(self.X_train.columns)
        importance_df = pd.DataFrame({"feature": features, "importance": importance}).sort_values(
            "importance", ascending=False
        )

        return importance_df

    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about the best model."""
        if self.best_model_name is None or self.best_model is None:
            return {}

        metrics = self.results.get(self.best_model_name, {})
        insights: Dict[str, Any] = {
            "best_model_name": self.best_model_name,
            "best_model_type": type(self.best_model).__name__,
            "metrics": metrics,
            "feature_importance": self.get_feature_importance(self.best_model),
        }

        # Add model-specific attributes
        if hasattr(self.best_model, "n_features_in_"):
            insights["n_features"] = getattr(self.best_model, "n_features_in_")

        if hasattr(self.best_model, "feature_names_in_"):
            insights["feature_names"] = list(getattr(self.best_model, "feature_names_in_"))

        return insights

    def save_models(self, output_dir: str = "models") -> None:
        """Save all trained models."""
        ensure_dir(output_dir)
        timestamp = get_timestamp()

        assert self.best_model is not None, "No best model to save."

        # Save best model
        best_model_path = os.path.join(output_dir, f"best_model_{timestamp}.joblib")
        save_model(self.best_model, best_model_path)

        # Save all models
        for name, model in self.trained_models.items():
            safe_name = name.replace(" ", "_").lower()
            model_path = os.path.join(output_dir, f"{safe_name}_{timestamp}.joblib")
            save_model(model, model_path)

        print("\nAll models saved to: {}/".format(output_dir))

    def save_results(self, output_dir: str = "outputs") -> None:
        """Save training results and evaluation metrics."""
        ensure_dir(output_dir)
        timestamp = get_timestamp()

        # ensure data is present
        assert (
            self.X_train is not None and self.X_test is not None
        ), "Data not set. Call train/test split first."
        assert (
            self.best_model_name is not None and self.best_model is not None
        ), "No trained model present."

        # Results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_csv = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
        results_df.to_csv(results_csv)

        # Full evaluation report
        report = {
            "timestamp": timestamp,
            "best_model": self.best_model_name,
            "best_model_type": type(self.best_model).__name__,
            "metrics": self.results[self.best_model_name],
            "all_models": self.results,
            "data_shape": {
                "train": len(self.X_train),
                "test": len(self.X_test),
                "features": self.X_train.shape[1],
            },
        }
        filename = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        save_json(report, filename)

        print("\nResults saved to: {}/".format(output_dir))


# ============================================
# VISUALIZATION
# ============================================
class ModelVisualizer:
    """Professional model visualization utilities."""

    @staticmethod
    def plot_model_comparison(
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot model comparison bar chart."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = list(results.keys())
        r2_scores = [results[m]["r2"] for m in models]
        rmse_scores = [results[m]["rmse"] for m in models]
        mae_scores = [results[m]["mae"] for m in models]

        # R² plot
        ax = axes[0]
        colors = ["green" if i == 0 else "skyblue" for i in range(len(models))]
        ax.bar(models, r2_scores, color=colors)
        ax.set_title("R² Score", fontsize=12)
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)

        # RMSE plot
        ax = axes[1]
        ax.bar(models, rmse_scores, color=colors)
        ax.set_title("RMSE (ETB)", fontsize=12)
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=45)

        # MAE plot
        ax = axes[2]
        ax.bar(models, mae_scores, color=colors)
        ax.set_title("MAE (ETB)", fontsize=12)
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Model comparison plot saved to: {}".format(save_path))

        plt.show()

    @staticmethod
    def plot_predictions(
        y_true: pd.Series,
        y_pred: pd.Series,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("Actual Price (ETB)")
        ax.set_ylabel("Predicted Price (ETB)")
        ax.set_title("Actual vs Predicted")

        # Residuals plot
        ax = axes[1]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color="r", linestyle="--", lw=2)
        ax.set_xlabel("Predicted Price (ETB)")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Predictions plot saved to: {}".format(save_path))

        plt.show()

    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance."""
        importance_df = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color="skyblue",
        )
        ax.set_xlabel("Importance")
        ax.set_title("Top {} Feature Importance".format(top_n))
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Feature importance plot saved to: {}".format(save_path))

        plt.show()


# ============================================
# MAIN PIPELINE
# ============================================
class TrainingPipeline:
    """Complete training pipeline orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.preprocessor: DataPreprocessor = DataPreprocessor()
        self.trainer: Optional[ModelTrainer] = None
        self.visualizer: ModelVisualizer = ModelVisualizer()
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None

    def run(
        self,
        data_source: Optional[str] = None,
        n_samples: int = 15000,
        test_size: float = 0.2,
        use_hyperparameter_tuning: bool = True,
        create_ensemble: bool = True,
        save_models: bool = True,
        generate_plots: bool = True,
        output_dir: str = "outputs",
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Parameters:
        -----------
        data_source : str, optional
            Path to CSV file. If None, generate synthetic data.
        n_samples : int
            Number of samples to generate if no data source provided.
        test_size : float
            Test split ratio.
        use_hyperparameter_tuning : bool
            Whether to perform hyperparameter tuning.
        create_ensemble : bool
            Whether to create an ensemble model.
        save_models : bool
            Whether to save models to disk.
        generate_plots : bool
            Whether to generate visualization plots.
        output_dir : str
            Output directory for results.

        Returns:
        --------
        dict
            Training results and insights.
        """
        print("ADDIS ABABA HOUSING PRICE PREDICTOR")
        print("   Professional Training Pipeline")
        print("=" * 60)

        # 1. Load or generate data
        print("\nLoading Data...")
        self.data = self._load_data(data_source, n_samples)
        print("   Loaded {:,} records".format(len(self.data)))

        # 2. Preprocess data
        print("\nPreprocessing Data...")
        self.X, self.y = self.preprocessor.fit_transform(self.data)
        print("   Features: {}".format(self.X.shape[1]))

        # 3. Split data
        print("\nSplitting Data...")
        self.trainer = ModelTrainer(cv_folds=5, n_iter=20)
        assert self.trainer is not None
        # Narrow types for mypy
        assert self.X is not None and self.y is not None
        self.trainer.train_test_split(self.X, self.y, test_size=test_size)

        # 4. Train models
        print("\nTraining Models...")
        self.trainer.train_all_models(use_hyperparameter_tuning)

        # 5. Create ensemble
        if create_ensemble:
            self.trainer.create_ensemble()

        # 6. Save models
        if save_models:
            self.trainer.save_models("models")
            self.preprocessor.save("models/preprocessor.pkl")

        # 7. Generate evaluation report
        print("\nGenerating Evaluation Report...")
        self.trainer.save_results(output_dir)

        # 8. Generate visualizations
        if generate_plots:
            print("\nGenerating Visualizations...")
            self._generate_visualizations(output_dir)

        # 9. Get insights
        insights = self.trainer.get_model_insights()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        if insights:
            print("Best Model: {}".format(insights.get("best_model_name")))
            metrics = insights.get("metrics", {})
            if metrics:
                print(" R² Score: {:.4f}".format(metrics.get("r2", 0.0)))
                print(" RMSE: {:,.2f} ETB".format(metrics.get("rmse", 0.0)))
                print(" MAE: {:,.2f} ETB".format(metrics.get("mae", 0.0)))
        print("\n Results saved to:")
        print("   - Models: models/")
        print("   - Reports: {}/".format(output_dir))

        return insights

    def _load_data(self, data_source: Optional[str], n_samples: int) -> pd.DataFrame:
        """Load data from source or generate synthetic data."""
        if data_source and os.path.exists(data_source):
            return pd.read_csv(data_source)
        else:
            if data_source:
                print("   File not found: {}".format(data_source))
            print("   Generating {:,} synthetic records...".format(n_samples))
            return generate_housing_data(n_samples=n_samples, seed=42)

    def _generate_visualizations(self, output_dir: str) -> None:
        """Generate all visualization plots."""
        # Ensure trainer and test data exist for type checkers
        assert self.trainer is not None, "Trainer has not been initialized."
        assert self.trainer.best_model is not None, "No best model available to plot."
        assert (
            self.trainer.X_test is not None and self.trainer.y_test is not None
        ), "Test data split missing."

        # Model comparison
        model_comp_path = os.path.join(output_dir, "model_comparison.png")
        self.visualizer.plot_model_comparison(
            self.trainer.results,
            save_path=model_comp_path,
        )

        # Predictions
        y_pred = self.trainer.best_model.predict(self.trainer.X_test)
        predictions_path = os.path.join(output_dir, "predictions.png")
        self.visualizer.plot_predictions(
            self.trainer.y_test,
            y_pred,
            save_path=predictions_path,
        )

        # Feature importance
        importance = self.trainer.get_feature_importance(self.trainer.best_model)
        if importance is not None:
            feat_imp_path = os.path.join(output_dir, "feature_importance.png")
            self.visualizer.plot_feature_importance(
                importance,
                top_n=10,
                save_path=feat_imp_path,
            )


# ============================================
# CLI ENTRY POINT
# ============================================
def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train housing price prediction models")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        help=("Path to CSV data file (if not provided, synthetic data will be generated)"),
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=15000,
        help="Number of synthetic samples to generate (default: 15000)",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble model creation",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    # Run pipeline
    pipeline = TrainingPipeline()
    pipeline.run(
        data_source=args.data,
        n_samples=args.samples,
        test_size=args.test_size,
        use_hyperparameter_tuning=not args.no_tuning,
        create_ensemble=not args.no_ensemble,
        save_models=True,
        generate_plots=not args.no_plots,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
