"""
File : utils/range_checker.py
"""

import pandas as pd
import logging

# Define acceptable ranges for relevant columns
RANGE_CHECKS = {
    "rating": (1, 5),           # Ratings should always be between 1 and 5
    "price": (0, None),         # Price should be non-negative
    "year": (2018, 2023),       # Only reviews from 2018 to 2023 are expected
    "rating_number": (1, None)  # Ratings count should be positive
}

def check_range(data: pd.DataFrame) -> (dict, bool):
    """
    Check if numeric columns are within their expected ranges.

    Parameters:
    data : DataFrame

    Returns:
    tuple:
        - Dictionary with column names as keys and lists of invalid row indices as values.
        - Boolean indicating if all ranges are valid (True if no issues, False otherwise).
    """
    invalid_rows = {}
    validation_passed = True  # Assume validation will pass

    # Iterate over the columns and their expected ranges
    for column, (min_val, max_val) in RANGE_CHECKS.items():
        if column in data.columns:
            # Check for values below the minimum value
            below_min_indices = data[data[column] < min_val].index.tolist()
            if below_min_indices:
                logging.error(f"Range check failed for '{column}'. Values below {min_val} at rows: {below_min_indices}")
                invalid_rows[column] = below_min_indices
                validation_passed = False

            # Check for values above the maximum value (if applicable)
            if max_val is not None:
                above_max_indices = data[data[column] > max_val].index.tolist()
                if above_max_indices:
                    logging.error(f"Range check failed for '{column}'. Values above {max_val} at rows: {above_max_indices}")
                    if column in invalid_rows:
                        invalid_rows[column].extend(above_max_indices)
                    else:
                        invalid_rows[column] = above_max_indices
                    validation_passed = False

    # Remove duplicate indices for each column
    invalid_rows = {col: list(set(indices)) for col, indices in invalid_rows.items()}

    if validation_passed:
        logging.info("Range check passed.")
    else:
        logging.error("Range check failed due to out-of-range values.")

    return invalid_rows, validation_passed

# Example usage
if __name__ == "__main__":
    # Example DataFrame for testing
    df = pd.read_csv("/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv")
    # Run range check
    invalid_rows, result = check_range(df)

    # Print results
    print(f"Invalid Rows: {invalid_rows}")
    print(f"Validation Passed: {result}")