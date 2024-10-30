"""
File : utils/privacy_compliance.py
"""

import pandas as pd
import logging
import re

# Columns to run PII checks on
PII_CHECK_COLUMNS = ["text", "user_id", "title"]

# Patterns for PII detection (with length validation for phone numbers)
PII_PATTERNS = {
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email pattern
    "Phone": r"\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4,10})\b",  
}

def is_valid_phone_number(phone_number: str) -> bool:
    """
    Validate phone number length. Ensures it is meaningful.
    Length should be between 10 to 15 characters (including separators).
    """
    digits = re.sub(r'\D', '', phone_number)  # Remove non-digit characters
    return 10 <= len(digits) <= 15  # Adjust based on your needs

def check_data_privacy(data: pd.DataFrame) -> (list, bool):
    """
    Check if specified columns contain personal information (PII).

    Parameters:
    data : DataFrame 

    Returns:
    tuple: 
        - List of row indices with potential PII.
        - Boolean indicating if data is compliant (True if no PII found, False otherwise).
    """
    pii_rows = []  # Store row indices with PII issues
    validation_passed = True  # Assume data is compliant initially

    # Iterate through specified columns to check for PII patterns
    for column in PII_CHECK_COLUMNS:
        if column not in data.columns:
            logging.warning(f"Column '{column}' not found in the data.")
            continue

        # Check for each PII pattern in the column
        for pii_type, pattern in PII_PATTERNS.items():
            pii_indices = data[data[column].astype(str).str.contains(pattern, regex=True)].index.tolist()

            for idx in pii_indices:
                text_content = data.at[idx, column]

                if pii_type == "Phone" and not is_valid_phone_number(text_content):
                    logging.info(f"Skipping invalid phone number at row {idx}. Text: {text_content}")
                    continue  # Skip if phone number is invalid

                logging.error(f"Potential {pii_type} found in '{column}' column at row {idx}. Text: {text_content}")
                pii_rows.append(idx)
                validation_passed = False

    # Remove duplicate row indices (if any)
    pii_rows = list(set(pii_rows))

    # Log the final result
    if validation_passed:
        logging.info("Data privacy compliance passed.")
    else:
        logging.error("Data privacy compliance failed due to PII found.")

    return pii_rows, validation_passed

# Example usage:
if __name__ == "__main__":
    # Example DataFrame for testing
    data = pd.read_csv("/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv")
    # Run privacy compliance check
    pii_rows, result = check_data_privacy(data)

    # Print results
    print(f"PII Rows: {pii_rows}")
    print(f"Validation Passed: {result}")
