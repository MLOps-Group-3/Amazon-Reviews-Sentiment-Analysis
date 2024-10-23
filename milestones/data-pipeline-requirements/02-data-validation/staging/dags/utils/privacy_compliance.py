"""
File : utils/privacy_compliance.py
"""

import pandas as pd
import logging

# Columns to run PII checks on
PII_CHECK_COLUMNS = ["text", "user_id", "title"]  

# Patterns for PII detection
PII_PATTERNS = {
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email pattern
    "Phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US Phone number pattern
}

def check_data_privacy(data: pd.DataFrame) -> bool:
    """
    Check if specified columns contain personal information.

    Parameters:
    data : DataFrame 

    Returns:
    bool: True if data is compliant, False otherwise.
    """
    for column in PII_CHECK_COLUMNS:
        if column not in data.columns:
            logging.warning(f"Column '{column}' not found in the data.")
            continue

        # Check for each PII pattern
        for pii_type, pattern in PII_PATTERNS.items():
            if data[column].astype(str).str.contains(pattern, regex=True).any():
                logging.error(f"Potential {pii_type} found in '{column}' column.")
                return False

    logging.info("Data privacy compliance passed.")
    return True


# Example Test :
#check_data_privacy(pd.read_csv("/home/shirish/Desktop/mlops/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv"))