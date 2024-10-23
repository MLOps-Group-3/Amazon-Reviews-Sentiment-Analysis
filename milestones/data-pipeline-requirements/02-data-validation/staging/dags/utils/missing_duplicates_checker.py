"""
utils/missing_duplicates_checker.py
"""

import pandas as pd
import logging

# Critical columns that must not contain missing values
CRITICAL_COLUMNS = ["asin", "user_id", "rating", "timestamp","text"]

def find_missing_and_duplicates(data: pd.DataFrame) -> bool:
    """
    Check for missing values in all columns, with a focus on critical columns,
    and check for duplicate rows in the DataFrame.

    Parameters:
    data : DataFrame 

    Returns:
    bool: True if no issues are found, False otherwise.
    """
    validation_passed = True

    # 1. Check all columns for missing values
    for column in data.columns:
        if data[column].isnull().any():
            missing_count = data[column].isnull().sum()
            if column in CRITICAL_COLUMNS:
                logging.error(f"Critical column '{column}' has {missing_count} missing values.")
                validation_passed = False
            else:
                logging.warning(f"Column '{column}' has {missing_count} missing values.")

    # 2. Check for duplicate rows
    if data.duplicated().any():
        duplicate_count = data.duplicated().sum()
        logging.error(f"Found {duplicate_count} duplicate rows.")
        validation_passed = False

    if validation_passed:
        logging.info("No critical missing values or duplicate rows found.")
    else:
        logging.error("Validation failed due to missing values or duplicates.")

    return validation_passed
