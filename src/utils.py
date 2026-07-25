"""
Utility functions for data processing, analysis, and model management
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# MODEL PERSISTENCE UTILITIES
# ============================================


def save_model(model, filepath: str) -> None:
    """Save a trained model using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")


def load_model(filepath: str):
    """Load a saved model using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def save_predictions(y_true, y_pred, output_path: str) -> None:
    """Save predictions with actual values and errors to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "error": y_true - y_pred,
            "abs_error": np.abs(y_true - y_pred),
        }
    )
    results.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")


def get_model_info(model) -> Dict:
    """Extract useful information from a trained model."""
    return {
        "type": type(model).__name__,
        "features": getattr(model, "n_features_in_", "unknown"),
        "classes": getattr(model, "classes_", None),
        "coef": getattr(model, "coef_", None),
        "intercept": getattr(model, "intercept_", None),
        "feature_names": getattr(model, "feature_names_in_", None),
    }


# ============================================
# FILE AND DIRECTORY UTILITIES
# ============================================


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)
    print(f"✅ JSON saved to: {filepath}")


def load_json(filepath: str) -> Dict:
    """Load data from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(obj, filepath: str) -> None:
    """Save object using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Pickle saved to: {filepath}")


def load_pickle(filepath: str):
    """Load object from pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ============================================
# DATA PROCESSING UTILITIES
# ============================================


def basic_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive data summary."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_stats": df.describe().to_dict(),
        "categorical_stats": {
            col: df[col].value_counts().to_dict()
            for col in df.select_dtypes(include=["object"]).columns
        },
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing outliers and invalid values."""
    df_clean = df.copy()

    if "price" in df_clean.columns:
        df_clean = df_clean[(df_clean["price"] >= 100000) & (df_clean["price"] <= 200000000)]

    if "area" in df_clean.columns:
        df_clean = df_clean[(df_clean["area"] >= 10) & (df_clean["area"] <= 1500)]

    if "bedrooms" in df_clean.columns:
        df_clean = df_clean[(df_clean["bedrooms"] >= 1) & (df_clean["bedrooms"] <= 10)]

    if "bathrooms" in df_clean.columns:
        df_clean = df_clean[(df_clean["bathrooms"] >= 1) & (df_clean["bathrooms"] <= 8)]

    return df_clean.reset_index(drop=True)


def get_price_per_sqm(df: pd.DataFrame) -> pd.Series:
    """Calculate price per square meter."""
    if "price" not in df.columns or "area" not in df.columns:
        raise ValueError("Dataset must contain 'price' and 'area' columns")
    return df["price"] / df["area"]


def split_by_location(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Filter dataset by specific location."""
    if "location" not in df.columns:
        raise ValueError("Dataset must contain 'location' column")
    return df[df["location"] == location].copy()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ============================================
# MODEL EVALUATION UTILITIES
# ============================================


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate common regression metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae_percent": (mae / np.mean(y_true)) * 100,
    }


def save_evaluation_report(
    metrics: Dict,
    output_path: str,
    model_name: Optional[str] = None,
) -> None:
    """Save evaluation metrics to JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name or "Unnamed Model",
        "metrics": metrics,
    }
    save_json(report, output_path)


# ============================================
# TIMESTAMP UTILITIES
# ============================================


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_date() -> str:
    """Get current date as string."""
    return datetime.now().strftime("%Y-%m-%d")


def get_file_timestamp() -> str:
    """Get timestamp suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================
# DATA VALIDATION UTILITIES
# ============================================


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Validate that all required columns exist in dataframe."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary of missing values in dataset."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    summary = pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
    return summary[summary["missing_count"] > 0].sort_values("missing_count", ascending=False)
