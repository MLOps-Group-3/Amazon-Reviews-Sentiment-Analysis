"""
File : utils/range_checker.py
"""

## NEED  TO CHECK THE NULL VALUES

import pandas as pd
import logging

# Define acceptable ranges for relevant columns
RANGE_CHECKS = {
    "rating": (1,5),           # Ratings should alwyas be between 1 and 5
    "price": (0, None),        # Price should be non-negative and reasonable
    "year": (2018, 2023),        # Only reviews from 2018 to 2023 are expected in our case
    "rating_number": (1,None)  # Ratings count in a reasonable range
}

def check_range(data: pd.DataFrame) -> bool:
    """
    Check if numeric columns are within their expected ranges.

    Parameters:
    data : DataFrame

    Returns:
    bool: True if all ranges are valid, False otherwise.
    """
    for column, (min_val, max_val) in RANGE_CHECKS.items():
        if column in data.columns:
            # Check if values are above the minimum value
            if (data[column] < min_val).any():
                invalid_rows = data[data[column] < min_val]
                logging.error(f"Range check failed for '{column}'. Values below {min_val}:\n{invalid_rows}")
                return False

            # Check if values exceed the maximum value (only if max_val is not None)
            if max_val is not None and (data[column] > max_val).any():
                invalid_rows = data[data[column] > max_val]
                logging.error(f"Range check failed for '{column}'. Values above {max_val}:\n{invalid_rows}")
                return False

    logging.info("Range check passed.")
    return True



# Example Test :
#check_range(pd.read_csv("/home/shirish/Desktop/mlops/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv"))
