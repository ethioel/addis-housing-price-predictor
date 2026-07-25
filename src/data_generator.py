"""
Addis Housing Data Generator
Synthetic dataset based on Jiji.com.et property listings for Addis Ababa
"""

import random
from typing import Optional

import numpy as np
import pandas as pd

# ============================================
# CONSTANTS - Based on Jiji.com Data
# ============================================

LOCATIONS = {
    "Bole": {"weight": 0.610, "multiplier": 1.3, "base_price": 65000},
    "Yeka": {"weight": 0.211, "multiplier": 1.0, "base_price": 50000},
    "Akaky Kaliti": {"weight": 0.058, "multiplier": 0.7, "base_price": 35000},
    "Nifas Silk-Lafto": {"weight": 0.050, "multiplier": 0.85, "base_price": 42000},
    "Kirkos": {"weight": 0.027, "multiplier": 1.1, "base_price": 55000},
    "Arada": {"weight": 0.013, "multiplier": 0.9, "base_price": 45000},
    "Lideta": {"weight": 0.012, "multiplier": 0.7, "base_price": 35000},
    "Kolfe Keranio": {"weight": 0.009, "multiplier": 0.75, "base_price": 38000},
    "Gullele": {"weight": 0.006, "multiplier": 0.7, "base_price": 35000},
    "Addis Ketema": {"weight": 0.004, "multiplier": 0.7, "base_price": 35000},
}

PROPERTY_TYPES = {
    "House": {"weight": 0.38, "area_min": 100, "area_max": 400},
    "Apartment": {"weight": 0.27, "area_min": 40, "area_max": 180},
    "Condo": {"weight": 0.17, "area_min": 50, "area_max": 150},
    "Villa": {"weight": 0.14, "area_min": 200, "area_max": 600},
    "Studio": {"weight": 0.04, "area_min": 20, "area_max": 50},
}

FURNISHING = ["furnished", "semi-furnished", "unfurnished"]
FURNISHING_WEIGHTS = [0.477, 0.168, 0.355]

CONDITIONS = [
    "Newly-Built",
    "Fairly Used",
    "Uncompleted Building",
    "Old",
    "Under Construction",
    "Renovated",
    "Off-Plan",
]
CONDITION_WEIGHTS = [0.682, 0.25, 0.022, 0.022, 0.013, 0.006, 0.005]

PROPERTY_MULTIPLIER = {
    "House": 1.0,
    "Apartment": 0.85,
    "Condo": 0.8,
    "Villa": 1.3,
    "Studio": 0.6,
}

# Probability constants (from Jiji data)
PROB_MAINROAD = 0.75
PROB_GUESTROOM = 0.25
PROB_BASEMENT = 0.15
PROB_HOTWATER = 0.35
PROB_AIRCONDITIONING = 0.55
PROB_PREFAREA = 0.30

# Price bounds
MIN_PRICE = 500_000
MAX_PRICE = 150_000_000
PRICE_NOISE_STD = 0.15


# ============================================
# GENERATOR FUNCTION
# ============================================


def generate_housing_data(
    n_samples: int = 15000,
    seed: Optional[int] = 42,
    include_condition: bool = True,
    include_property_type: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic Ethiopian housing dataset.

    Parameters:
    -----------
    n_samples : int, default=15000
        Number of records to generate
    seed : int or None, default=42
        Random seed for reproducibility (None for random)
    include_condition : bool, default=True
        Whether to include the 'condition' column
    include_property_type : bool, default=True
        Whether to include the 'property_type' column

    Returns:
    --------
    pd.DataFrame
        Generated dataset with housing features

    Examples:
    ---------
    >>> df = generate_housing_data(n_samples=100)
    >>> df.shape
    (100, 16)
    """
    # Set random seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = []
    locations = list(LOCATIONS.keys())
    location_weights = [LOCATIONS[loc]["weight"] for loc in locations]
    property_types = list(PROPERTY_TYPES.keys())
    property_weights = [PROPERTY_TYPES[pt]["weight"] for pt in property_types]

    for _ in range(n_samples):
        # ========================================
        # 1. Select Location & Property Type
        # ========================================
        location = random.choices(locations, weights=location_weights, k=1)[0]
        prop_type = random.choices(property_types, weights=property_weights, k=1)[0]

        # ========================================
        # 2. Generate Area
        # ========================================
        area_min = PROPERTY_TYPES[prop_type]["area_min"]
        area_max = PROPERTY_TYPES[prop_type]["area_max"]
        area = int(np.random.triangular(area_min, (area_min + area_max) / 2, area_max))

        # ========================================
        # 3. Generate Bedrooms
        # ========================================
        bedrooms = _generate_bedrooms(area, prop_type)

        # ========================================
        # 4. Generate Bathrooms
        # ========================================
        bathrooms = _generate_bathrooms(bedrooms)

        # ========================================
        # 5. Generate Stories
        # ========================================
        stories = _generate_stories(prop_type)

        # ========================================
        # 6. Generate Categorical Features
        # ========================================
        mainroad = "yes" if random.random() < PROB_MAINROAD else "no"
        guestroom = "yes" if random.random() < PROB_GUESTROOM else "no"
        basement = "yes" if random.random() < PROB_BASEMENT else "no"
        hotwaterheating = "yes" if random.random() < PROB_HOTWATER else "no"
        airconditioning = "yes" if random.random() < PROB_AIRCONDITIONING else "no"
        prefarea = "yes" if random.random() < PROB_PREFAREA else "no"
        furnishingstatus = random.choices(FURNISHING, weights=FURNISHING_WEIGHTS, k=1)[0]
        condition = random.choices(CONDITIONS, weights=CONDITION_WEIGHTS, k=1)[0]
        parking = int(np.random.triangular(0, 2, 5))

        # ========================================
        # 7. Generate Price
        # ========================================
        price = _generate_price(area, location, prop_type, bedrooms, bathrooms)

        # ========================================
        # 8. Build Record
        # ========================================
        record = {
            "price": price,
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
            "location": location,
        }

        if include_condition:
            record["condition"] = condition

        if include_property_type:
            record["property_type"] = prop_type

        data.append(record)

    return pd.DataFrame(data)


# ============================================
# HELPER FUNCTIONS
# ============================================


def _generate_bedrooms(area: int, property_type: str) -> int:
    """Generate bedrooms based on area and property type."""
    if property_type in ["Apartment", "Condo"]:
        if area < 50:
            return 1
        if area < 80:
            return 2
        if area < 120:
            return 3
        if area < 160:
            return 4
        return 5
    if property_type == "Studio":
        return 1
    # House, Villa
    if area < 100:
        return 2
    if area < 150:
        return 3
    if area < 200:
        return 4
    if area < 300:
        return 5
    return 6


def _generate_bathrooms(bedrooms: int) -> int:
    """Generate bathrooms based on bedrooms."""
    if bedrooms == 1:
        return 1
    if bedrooms == 2:
        return random.choice([1, 2])
    if bedrooms == 3:
        return random.choice([2, 3])
    if bedrooms == 4:
        return random.choice([2, 3, 4])
    return random.choice([3, 4, 5])


def _generate_stories(property_type: str) -> int:
    """Generate number of stories based on property type."""
    if property_type == "Apartment":
        return random.choice([1, 2, 3, 4])
    if property_type in ["Condo", "Villa"]:
        return random.choice([1, 2, 3])
    if property_type == "House":
        return random.choice([1, 2])
    return 1  # Studio


def _generate_price(
    area: int,
    location: str,
    property_type: str,
    bedrooms: int,
    bathrooms: int,
) -> int:
    """Generate price in ETB based on property features."""
    location_multiplier = LOCATIONS[location]["multiplier"]
    base_price = LOCATIONS[location]["base_price"]
    property_multiplier = PROPERTY_MULTIPLIER[property_type]

    bedroom_bonus = (bedrooms - 1) * 0.05
    bathroom_bonus = (bathrooms - 1) * 0.03

    price = (
        area
        * base_price
        * location_multiplier
        * property_multiplier
        * (1 + bedroom_bonus + bathroom_bonus)
    )

    # Add random noise
    price = int(price * np.random.normal(1, PRICE_NOISE_STD))

    # Ensure price is within bounds
    return max(MIN_PRICE, min(MAX_PRICE, price))


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================


def generate_and_save(
    n_samples: int = 15000,
    output_path: str = "data/ethiopian_housing_data.csv",
    seed: Optional[int] = 42,
    include_condition: bool = True,
    include_property_type: bool = True,
) -> pd.DataFrame:
    """
    Generate dataset and save to CSV.
    """
    df = generate_housing_data(
        n_samples=n_samples,
        seed=seed,
        include_condition=include_condition,
        include_property_type=include_property_type,
    )

    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df):,} records and saved to: {output_path}")
    return df


def get_location_distribution(df: pd.DataFrame) -> pd.Series:
    """Get location distribution as percentages."""
    if "location" not in df.columns:
        raise ValueError("Dataset must contain 'location' column")
    return df["location"].value_counts(normalize=True).round(3)


def get_property_type_distribution(df: pd.DataFrame) -> pd.Series:
    """Get property type distribution as percentages."""
    if "property_type" not in df.columns:
        raise ValueError("Dataset must contain 'property_type' column")
    return df["property_type"].value_counts(normalize=True).round(3)


def print_summary(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the dataset."""
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)

    print(f"\n📈 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n📋 Columns:")
    for col in df.columns:
        dtype = df[col].dtype
        unique = df[col].nunique()
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100 if len(df) > 0 else 0
        print(f"  - {col}: {dtype} ({unique:,} unique, {missing} missing, {missing_pct:.1f}%)")

    print("\n📊 Numerical Statistics:")
    print(df.describe().round(2))

    if "location" in df.columns:
        print("\n📍 Location Distribution:")
        print(df["location"].value_counts(normalize=True).round(3))

    if "property_type" in df.columns:
        print("\n🏠 Property Type Distribution:")
        print(df["property_type"].value_counts(normalize=True).round(3))


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("🏠 Ethiopian Housing Data Generator")
    print("=" * 40)

    print("\n🚀 Generating dataset...")
    df = generate_housing_data(n_samples=15000, seed=42)

    print_summary(df)

    import os

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/ethiopian_housing_data.csv", index=False)
    print("\n✅ Data saved to: data/ethiopian_housing_data.csv")

    print("\n📁 File ready for your ML pipeline!")
