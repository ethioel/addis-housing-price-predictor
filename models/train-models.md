# Basic training (uses synthetic data)
python -m src.train

# Train with custom data
python -m src.train --data data/ethiopian_housing_data.csv

# Train with custom parameters
python -m src.train --samples 20000 (you can change the amonunt) --test-size 0.15 --output outputs/

# Disable hyperparameter tuning (faster)
python -m src.train --no-tuning

# Disable ensemble creation
python -m src.train --no-ensemble

# Disable plots (faster)
python -m src.train --no-plots

# See all options
python -m src.train --help
