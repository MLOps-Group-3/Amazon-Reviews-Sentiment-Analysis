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
    "year": (2018, 2023),        # Only reviews from these years are expected
    "rating_number": (1,None)  # Ratings count in a reasonable range
}

def check_range(data: pd.DataFrame) -> bool:
    """
    Check if numeric columns are within their expected ranges.

    Parameters:
    data (pd.DataFrame): DataFrame to validate.

    Returns:
    bool: True if all ranges are valid, False otherwise.
    """
    for column, (min_val, max_val) in RANGE_CHECKS.items():
        if column in data.columns:
            if not data[column].between(min_val, max_val).all():
                invalid_rows = data[~data[column].between(min_val, max_val)]
                logging.error(f"Range check failed for {column}. Invalid rows:\n{invalid_rows}")
                return False

    logging.info("Range check passed.")
    return True
