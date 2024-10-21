""" 
FILE : utils/schema_validation.py
"""

import pandas as pd
import logging

# Expected Schema
EXPECTED_SCHEMA = {
    "review_month": "int64",              # Integer representing month (1-12)
    "rating": "float64",                  # Product rating (e.g., 4.5)
    "parent_asin": "object",              # Parent product ASIN
    "asin": "object",                     # Product-specific ASIN
    "helpful_vote": "int64",              # Number of helpful votes
    "text": "object",                     # Review text
    "timestamp": "int64",                 # Unix timestamp
    "title": "object",                    # Review title
    "user_id": "object",                  # User identifier
    "verified_purchase": "bool",          # Whether the purchase was verified
    "review_date_timestamp": "object",    # Review date in string format
    "main_category": "object",            # Main category of the product
    "product_name": "object",             # Product name
    "categories": "object",               # Categories associated with the product
    "price": "float64",                   # Price of the product
    "average_rating": "float64",          # Average product rating
    "rating_number": "int64",             # Number of ratings received
    "year": "int64"                       # Year of the review
}

def validate_schema(data: pd.DataFrame) -> bool:
    """
    Comapre the exsiting schema to the New schema

    Parameters:
    data : DataFrame to validate.

    Returns:
    bool: True if the schema is valid, False otherwise.
    """
    
    logging.info("Schema validation Started ..")
    
    # Check for missing columns
    for column, expected_dtype in EXPECTED_SCHEMA.items():
        if column not in data.columns:
            logging.error(f"Missing column: {column}")
            return False
        
        # Check if the data type matches the expected type
        actual_dtype = str(data[column].dtype)
        if actual_dtype != expected_dtype:
            logging.error(
                f"Invalid type for column '{column}'. "
                f"Expected {expected_dtype}, found {actual_dtype}."
            )
            return False

    # Check for unexpected extra columns
    extra_columns = set(data.columns) - set(EXPECTED_SCHEMA.keys())
    if extra_columns:
        logging.warning(f"Unexpected columns found: {extra_columns}")

    logging.info("Schema validation passed.")
    return True
