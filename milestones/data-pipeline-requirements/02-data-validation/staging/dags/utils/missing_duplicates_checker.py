"""
utils/missing_duplicates_checker.py
"""

import pandas as pd
import logging

# Critical columns that must not contain missing values
CRITICAL_COLUMNS = ["asin", "user_id", "rating", "timestamp", "text"]

def find_missing_and_duplicates(data: pd.DataFrame) -> (list, list, bool):
    """
    Check for missing values and duplicate rows in the DataFrame,
    focusing on critical columns.

    Parameters:
    data : DataFrame 

    Returns:
    tuple: 
        - List of row indices with missing values across critical columns.
        - List of duplicate row indices.
        - Bool indicating if the data passed all checks (True if no issues, otherwise False).
    """
    validation_passed = True
    incomplete_rows = []  # Rows with missing values across critical columns
    duplicate_rows = []  # Rows identified as duplicates

    # 1. Identify rows with missing values in critical columns
    for idx, row in data[CRITICAL_COLUMNS].iterrows():
        if row.isnull().any():
            missing_columns = row[row.isnull()].index.tolist()
            logging.error(f"Row {idx} has missing values in critical columns: {missing_columns}")
            incomplete_rows.append(idx)
            validation_passed = False

    # 2. Checking for duplicate rows
    duplicate_indices = data[data.duplicated()].index.tolist()
    if duplicate_indices:
        logging.error(f"Found duplicate rows at indices: {duplicate_indices}")
        duplicate_rows.extend(duplicate_indices)
        validation_passed = False

    # Log final aggregated  results
    logging.info(f"Total incomplete rows: {len(incomplete_rows)}")
    logging.info(f"Total duplicate rows: {len(duplicate_rows)}")

    # Final validation status
    if validation_passed:
        logging.info("Data validation passed: No missing values in critical columns or duplicate rows.")
    else:
        logging.error("Data validation failed due to missing values or duplicates.")

    return incomplete_rows, duplicate_rows, validation_passed

# Example usage:
if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv")

    # Run the validation
    incomplete_rows, duplicate_rows, result = find_missing_and_duplicates(df)

    # Print the results
    print(f"Incomplete Rows: {incomplete_rows}")
    print(f"Duplicate Rows: {duplicate_rows}")
    print(f"Validation Passed: {result}")
