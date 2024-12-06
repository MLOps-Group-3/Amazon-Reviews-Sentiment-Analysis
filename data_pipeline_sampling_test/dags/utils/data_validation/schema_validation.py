"""
FILE : utils/schema_validation.py
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Expected Schema
EXPECTED_SCHEMA = {
    "review_month": ["int64"],              # Integer representing month (1-12)
    "rating": ["float64"],                  # Product rating (e.g., 4.5)
    "parent_asin": ["object"],              # Parent product ASIN
    "asin": ["object"],                     # Product-specific ASIN
    "helpful_vote": ["int64"],              # Number of helpful votes
    "text": ["object"],                     # Review text
    "timestamp": ["int64", "object"],       # Unix timestamp
    "title": ["object"],                    # Review title
    "user_id": ["object"],                  # User identifier
    "verified_purchase": ["bool"],          # Whether the purchase was verified
    "review_date_timestamp": ["object"],    # Review date in string format
    "main_category": ["object"],            # Main category of the product
    "product_name": ["object"],             # Product name
    "categories": ["object"],               # Categories associated with the product
    "price": ["float64"],                   # Price of the product
    "average_rating": ["float64"],          # Average product rating
    "rating_number": ["int64"],             # Number of ratings received
    "year": ["int64"]                       # Year of the review
}

def validate_schema(data: pd.DataFrame) -> bool:
    """
    Validate and convert the schema of the given DataFrame.

    Parameters:
    data : DataFrame

    Returns:
    bool: True if the schema is valid, False otherwise.
    """
    logging.info("Schema validation and conversion started.")

    for column, expected_dtypes in EXPECTED_SCHEMA.items():
        if column not in data.columns:
            logging.error(f"Missing column: {column}")
            return False

        actual_dtype = str(data[column].dtype)
        if actual_dtype not in expected_dtypes:
            logging.warning(
                f"Invalid type for column '{column}'. "
                f"Expected {expected_dtypes}, found {actual_dtype}. "
                f"Attempting conversion."
            )
            # Attempt conversion to one of the expected data types
            converted = False
            for target_dtype in expected_dtypes:
                try:
                    if target_dtype == "bool":
                        data[column] = data[column].astype(target_dtype)
                    elif target_dtype == "int64":
                        data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0).astype(target_dtype)
                    elif target_dtype == "float64":
                        data[column] = pd.to_numeric(data[column], errors='coerce').astype(target_dtype)
                    elif target_dtype == "object":
                        data[column] = data[column].astype(str)
                    else:
                        data[column] = data[column].astype(target_dtype)
                    converted = True
                    logging.info(f"Successfully converted column '{column}' to {target_dtype}.")
                    break
                except Exception as e:
                    logging.error(f"Failed to convert column '{column}' to {target_dtype}. Error: {e}")
            
            if not converted:
                logging.error(f"Unable to convert column '{column}' to any of the expected types {expected_dtypes}.")
                return False

    # Check for unexpected extra columns
    extra_columns = set(data.columns) - set(EXPECTED_SCHEMA.keys())
    if extra_columns:
        logging.warning(f"Unexpected columns found: {extra_columns}")

    logging.info("Schema validation and conversion passed.")
    return True

# Example usage
if __name__ == "__main__":
    # Example DataFrame for testing
    df = pd.read_csv("/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv")

    result = validate_and_convert_schema(df)

    # Print results
    print(f"Validation Passed: {result}")
