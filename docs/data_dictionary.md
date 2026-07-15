# 📊 Data Dictionary

Complete feature documentation for the Addis Ababa housing dataset.

---

## Numeric Features

### price
- **Type:** Integer
- **Description:** Property price in Ethiopian Birr (ETB)
- **Range:** 500,000 - 150,000,000 ETB
- **Distribution:** Log-normal (right-skewed)
- **Generation Method:** `area × base_price × location_multiplier × property_multiplier × (1 + bedroom_bonus + bathroom_bonus) + noise`
- **Example:** `21,067,153`

### area
- **Type:** Integer
- **Description:** Property area in square meters
- **Range:** 20 - 600 sqm
- **Distribution:** Triangular (varies by property type)
- **Generation Method:** `np.random.triangular(area_min, (area_min + area_max) / 2, area_max)`
- **Example:** `229`

### bedrooms
- **Type:** Integer
- **Description:** Number of bedrooms
- **Range:** 1 - 6
- **Distribution:** Based on area and property type
- **Example:** `5`

### bathrooms
- **Type:** Integer
- **Description:** Number of bathrooms
- **Range:** 1 - 5
- **Distribution:** Based on bedrooms
- **Example:** `4`

### stories
- **Type:** Integer
- **Description:** Number of stories/floors
- **Range:** 1 - 4
- **Distribution:** Varies by property type
- **Example:** `2`

### parking
- **Type:** Integer
- **Description:** Number of parking spaces
- **Range:** 0 - 5
- **Distribution:** Triangular (0, 2, 5)
- **Example:** `3`

---

## Categorical Features

### mainroad
- **Type:** Categorical
- **Description:** Access to main road
- **Values:** `yes`, `no`
- **Probability:** 75% yes
- **Example:** `yes`

### guestroom
- **Type:** Categorical
- **Description:** Has guest room
- **Values:** `yes`, `no`
- **Probability:** 25% yes
- **Example:** `no`

### basement
- **Type:** Categorical
- **Description:** Has basement
- **Values:** `yes`, `no`
- **Probability:** 15% yes
- **Example:** `no`

### hotwaterheating
- **Type:** Categorical
- **Description:** Has hot water heating
- **Values:** `yes`, `no`
- **Probability:** 35% yes
- **Example:** `yes`

### airconditioning
- **Type:** Categorical
- **Description:** Has air conditioning
- **Values:** `yes`, `no`
- **Probability:** 55% yes
- **Example:** `yes`

### prefarea
- **Type:** Categorical
- **Description:** Preferred area designation
- **Values:** `yes`, `no`
- **Probability:** 30% yes
- **Example:** `no`

### furnishingstatus
- **Type:** Categorical
- **Description:** Furnishing status
- **Values:** `furnished`, `semi-furnished`, `unfurnished`
- **Distribution:** 47.7% furnished, 16.8% semi-furnished, 35.5% unfurnished
- **Example:** `furnished`

### condition
- **Type:** Categorical
- **Description:** Property condition
- **Values:** `Newly-Built`, `Fairly Used`, `Uncompleted Building`, `Old`, `Under Construction`, `Renovated`, `Off-Plan`
- **Distribution:** 68.2% Newly-Built, 25% Fairly Used, others < 2.2%
- **Example:** `Newly-Built`

### location
- **Type:** Categorical
- **Description:** Sub-city in Addis Ababa
- **Values:** `Bole`, `Yeka`, `Akaky Kaliti`, `Nifas Silk-Lafto`, `Kirkos`, `Arada`, `Lideta`, `Kolfe Keranio`, `Gullele`, `Addis Ketema`
- **Distribution:** See Location Distribution table below
- **Example:** `Bole`

### property_type
- **Type:** Categorical
- **Description:** Type of property
- **Values:** `House`, `Apartment`, `Condo`, `Villa`, `Studio`
- **Distribution:** 38% House, 27% Apartment, 17% Condo, 14% Villa, 4% Studio
- **Example:** `House`

---

## Location Distribution

| Sub-city | Percentage | Price Multiplier | Base Price (ETB/sqm) |
|----------|------------|------------------|---------------------|
| Bole | 61.0% | 1.30 | 65,000 |
| Yeka | 21.1% | 1.00 | 50,000 |
| Akaky Kaliti | 5.8% | 0.70 | 35,000 |
| Nifas Silk-Lafto | 5.0% | 0.85 | 42,000 |
| Kirkos | 2.7% | 1.10 | 55,000 |
| Arada | 1.3% | 0.90 | 45,000 |
| Lideta | 1.2% | 0.70 | 35,000 |
| Kolfe Keranio | 0.9% | 0.75 | 38,000 |
| Gullele | 0.6% | 0.70 | 35,000 |
| Addis Ketema | 0.4% | 0.70 | 35,000 |

---

## Property Type Distribution

| Property Type | Percentage | Area Range (sqm) |
|---------------|------------|------------------|
| House | 38.0% | 100 - 400 |
| Apartment | 27.0% | 40 - 180 |
| Condo | 17.0% | 50 - 150 |
| Villa | 14.0% | 200 - 600 |
| Studio | 4.0% | 20 - 50 |

---

## Derived Features (for modeling)

### price_per_sqm
- **Type:** Float
- **Description:** Price per square meter
- **Formula:** `price / area`
- **Example:** `92,000.00`

### location_encoded
- **Type:** Integer
- **Description:** Location as numerical encoding
- **Range:** 0 - 9
- **Example:** `0` (Bole)

### property_type_encoded
- **Type:** Integer
- **Description:** Property type as numerical encoding
- **Range:** 0 - 4
- **Example:** `0` (House)

---

## Data Quality Notes

- **Missing Values:** None - all fields are generated with complete data
- **Outliers:** Prices are bounded between 500,000 and 150,000,000 ETB
- **Reproducibility:** Use `seed` parameter for reproducible results
- **Distribution Match:** All distributions match manually extracted Jiji.com data
