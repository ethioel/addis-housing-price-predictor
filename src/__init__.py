"""
Addis Ababa Housing Price Predictor
Synthetic data generator for Ethiopian real estate
"""

__version__ = "1.0.0"
__author__ = "Samuel Kahsay"

from .data_generator import (
    generate_and_save,
    generate_housing_data,
    get_location_distribution,
    get_property_type_distribution,
    print_summary,
)

# Import utilities (explicitly)
from .utils import (
    basic_data_summary,
    clean_dataset,
    ensure_dir,
    get_categorical_columns,
    get_date,
    get_file_timestamp,
    get_missing_summary,
    get_model_info,
    get_numeric_columns,
    get_price_per_sqm,
    get_timestamp,
    load_json,
    load_model,
    load_pickle,
    regression_metrics,
    save_evaluation_report,
    save_json,
    save_model,
    save_pickle,
    save_predictions,
    split_by_location,
    validate_required_columns,
)

__all__ = [
    # Data generator
    "generate_housing_data",
    "generate_and_save",
    "print_summary",
    "get_location_distribution",
    "get_property_type_distribution",
    # Utilities
    "save_model",
    "load_model",
    "save_predictions",
    "get_model_info",
    "regression_metrics",
    "save_evaluation_report",
    "clean_dataset",
    "basic_data_summary",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "get_timestamp",
    "get_date",
    "get_file_timestamp",
    "validate_required_columns",
    "get_missing_summary",
    "get_price_per_sqm",
    "split_by_location",
    "get_categorical_columns",
    "get_numeric_columns",
]
