"""
Unit tests for housing data generator
"""

import pandas as pd
import pytest

from src.data_generator import (
    generate_and_save,
    generate_housing_data,
    get_location_distribution,
    get_property_type_distribution,
)


class TestHousingDataGenerator:
    """Test suite for housing data generator."""

    def test_generate_housing_data_shape(self):
        """Test that dataset has expected shape."""
        df = generate_housing_data(n_samples=100)
        assert len(df) == 100
        assert df.shape[1] >= 14

    def test_generate_housing_data_columns(self):
        """Test that all required columns are present."""
        required_cols = [
            "price",
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "parking",
            "prefarea",
            "furnishingstatus",
            "location",
        ]
        df = generate_housing_data(n_samples=10)
        for col in required_cols:
            assert col in df.columns

    def test_generate_housing_data_price_range(self):
        """Test that prices are within realistic bounds."""
        df = generate_housing_data(n_samples=1000)
        assert df["price"].min() >= 500000
        assert df["price"].max() <= 150000000
        assert df["price"].mean() > 1000000

    def test_generate_housing_data_area_range(self):
        """Test that areas are within realistic bounds."""
        df = generate_housing_data(n_samples=1000)
        assert df["area"].min() >= 20
        assert df["area"].max() <= 600
        assert df["area"].mean() > 50

    def test_generate_housing_data_bedroom_range(self):
        """Test that bedrooms are within realistic bounds."""
        df = generate_housing_data(n_samples=1000)
        assert df["bedrooms"].min() >= 1
        assert df["bedrooms"].max() <= 6

    def test_generate_housing_data_bathroom_range(self):
        """Test that bathrooms are within realistic bounds."""
        df = generate_housing_data(n_samples=1000)
        assert df["bathrooms"].min() >= 1
        assert df["bathrooms"].max() <= 5

    def test_generate_housing_data_stories_range(self):
        """Test that stories are within realistic bounds."""
        df = generate_housing_data(n_samples=1000)
        assert df["stories"].min() >= 1
        assert df["stories"].max() <= 4

    def test_generate_housing_data_parking_range(self):
        """Test that parking spaces are within realistic bounds."""
        df = generate_housing_data(n_samples=1000)
        assert df["parking"].min() >= 0
        assert df["parking"].max() <= 5

    def test_generate_housing_data_location_distribution(self):
        """Test that location distribution matches expected weights."""
        df = generate_housing_data(n_samples=5000)
        loc_dist = df["location"].value_counts(normalize=True).round(2)
        assert loc_dist["Bole"] > 0.50
        assert "Bole" in loc_dist.index
        assert len(loc_dist) == 10

    def test_generate_housing_data_property_type_distribution(self):
        """Test that property type distribution is reasonable."""
        df = generate_housing_data(n_samples=5000, include_property_type=True)
        prop_dist = df["property_type"].value_counts(normalize=True).round(2)
        assert prop_dist["House"] > 0.30
        assert "House" in prop_dist.index
        assert len(prop_dist) == 5

    def test_generate_housing_data_categorical_values(self):
        """Test that categorical columns have expected values."""
        df = generate_housing_data(n_samples=100)

        binary_cols = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]
        for col in binary_cols:
            assert set(df[col].unique()).issubset({"yes", "no"})

        assert set(df["furnishingstatus"].unique()).issubset(
            {"furnished", "semi-furnished", "unfurnished"}
        )

    def test_generate_housing_data_reproducible(self):
        """Test that seed produces reproducible results."""
        df1 = generate_housing_data(n_samples=100, seed=42)
        df2 = generate_housing_data(n_samples=100, seed=42)
        assert df1["price"].equals(df2["price"])
        assert df1["area"].equals(df2["area"])
        assert df1["bedrooms"].equals(df2["bedrooms"])

    def test_generate_housing_data_no_seed(self):
        """Test that no seed produces different results."""
        df1 = generate_housing_data(n_samples=100, seed=42)
        df2 = generate_housing_data(n_samples=100, seed=None)
        assert not df1["price"].equals(df2["price"])

    def test_generate_housing_data_condition_column(self):
        """Test condition column inclusion/exclusion."""
        df_with = generate_housing_data(n_samples=10, include_condition=True)
        assert "condition" in df_with.columns

        df_without = generate_housing_data(n_samples=10, include_condition=False)
        assert "condition" not in df_without.columns

    def test_generate_housing_data_property_type_column(self):
        """Test property_type column inclusion/exclusion."""
        df_with = generate_housing_data(n_samples=10, include_property_type=True)
        assert "property_type" in df_with.columns

        df_without = generate_housing_data(n_samples=10, include_property_type=False)
        assert "property_type" not in df_without.columns

    def test_generate_and_save(self, tmp_path):
        """Test generate_and_save function."""
        output_file = tmp_path / "test_data.csv"
        df = generate_and_save(n_samples=50, output_path=str(output_file), seed=42)
        assert output_file.exists()
        assert len(df) == 50

    def test_get_location_distribution(self):
        """Test location distribution function."""
        df = generate_housing_data(n_samples=100)
        dist = get_location_distribution(df)
        assert isinstance(dist, pd.Series)
        assert sum(dist) == 1.0
        assert "Bole" in dist.index

    def test_get_property_type_distribution(self):
        """Test property type distribution function."""
        df = generate_housing_data(n_samples=100, include_property_type=True)
        dist = get_property_type_distribution(df)
        assert isinstance(dist, pd.Series)
        assert sum(dist) == 1.0
        assert "House" in dist.index

    def test_generate_housing_data_no_missing(self):
        """Test that there are no missing values."""
        df = generate_housing_data(n_samples=100)
        assert df.isnull().sum().sum() == 0

    def test_generate_housing_data_dtypes(self):
        """Test that columns have correct data types."""
        df = generate_housing_data(n_samples=10)

        numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
        for col in numeric_cols:
            assert df[col].dtype in ["int64", "int32", "float64"]

        string_cols = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus",
            "location",
        ]
        for col in string_cols:
            # Check if dtype is string-like (object or StringDtype)
            assert pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
