"""
utils/anomaly_detector.py
"""

import pandas as pd
import logging
import json
from scipy.stats import zscore

# Configure logging to store logs in a file with JSON format
log_file_path = "anomaly_logs.json"
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
    level=logging.DEBUG  # Set to DEBUG to capture detailed logs
)
logger = logging.getLogger(__name__)

# Expected values for categorical fields
EXPECTED_MAIN_CATEGORIES = {"Electronics", "Books", "Clothing", "Beauty", "Toys"}
EXPECTED_VERIFIED_PURCHASE_VALUES = {"Y", "N"}
EXPECTED_CATEGORIES = {"Gadget", "Novel", "Apparel", "Cosmetics", "Games"}  # Sample expected categories

def detect_anomalies(data: pd.DataFrame) -> bool:
    """
    Detects anomalies in the DataFrame, including outliers, future dates, and unexpected categorical values.

    Parameters:
    data : DataFrame

    Returns:
    bool: True if no anomalies are found, False otherwise.
    """
    
    anomalies_found = False
    logging.info("Starting anomaly detection.")
    logging.debug(f"Data columns available for analysis: {list(data.columns)}")

    # 1. Outlier detection using z-score for numeric columns
    numeric_columns = ["rating", "helpful_vote", "price", "average_rating", "rating_number"]
    for column in numeric_columns:
        if column in data.columns:
            logger.info(f"Checking for outliers in '{column}' column.")
            column_data = data[column]
            logger.debug(f"Column '{column}' - data type: {column_data.dtype}, number of entries: {len(column_data)}")
            z_scores = zscore(column_data.fillna(0))  # fillna to avoid errors with missing data
            outliers = data[abs(z_scores) > 3]  # Define outliers as values with |z-score| > 3
            outlier_count = len(outliers)
            logger.debug(f"Outlier detection in '{column}' complete. Total outliers found: {outlier_count}")
            if outlier_count > 0:
                logger.warning(json.dumps({
                    "function": "detect_anomalies",
                    "issue": "outliers",
                    "column": column,
                    "outlier_count": outlier_count,
                    "message": f"Detected {outlier_count} outliers in '{column}' column."
                }))
                anomalies_found = True
            else:
                logger.info(f"No outliers detected in '{column}' column.")
        else:
            logger.debug(f"Skipping '{column}' as it is not present in the data.")

    # 2. Temporal anomalies (e.g., future dates in timestamps)
    if "review_date_timestamp" in data.columns:
        logger.info("Checking for future dates in 'review_date_timestamp' column.")
        logger.debug("Converting 'review_date_timestamp' to datetime format.")
        data["review_date_timestamp"] = pd.to_datetime(data["review_date_timestamp"], errors="coerce")
        future_dates = data[data["review_date_timestamp"] > pd.Timestamp.now()]
        future_date_count = len(future_dates)
        logger.debug(f"Future date check complete. Entries with future dates: {future_date_count}")
        if future_date_count > 0:
            logger.error(json.dumps({
                "function": "detect_anomalies",
                "issue": "future_dates",
                "column": "review_date_timestamp",
                "future_date_count": future_date_count,
                "message": f"{future_date_count} entries have future dates in 'review_date_timestamp'."
            }))
            anomalies_found = True
        else:
            logger.info("No future dates found in 'review_date_timestamp' column.")
    else:
        logger.debug("Skipping 'review_date_timestamp' as it is not present in the data.")

    # 3. Category/Label Consistency checks
    def check_category_consistency(column, expected_values):
        logger.info(f"Checking category consistency for '{column}' column.")
        logger.debug(f"Expected categories for '{column}': {expected_values}")
        unexpected_values = data[~data[column].isin(expected_values)]
        unexpected_count = len(unexpected_values)
        logger.debug(f"Category consistency check complete for '{column}'. Unexpected entries found: {unexpected_count}")
        if unexpected_count > 0:
            logger.error(json.dumps({
                "function": "detect_anomalies",
                "issue": "unexpected_category_value",
                "column": column,
                "unexpected_count": unexpected_count,
                "message": f"{unexpected_count} entries in '{column}' contain unexpected values."
            }))
            return True
        logger.info(f"All values in '{column}' column match expected categories.")
        return False

    if "main_category" in data.columns:
        logger.debug("Running category consistency check for 'main_category'.")
        anomalies_found |= check_category_consistency("main_category", EXPECTED_MAIN_CATEGORIES)
    else:
        logger.debug("Skipping 'main_category' as it is not present in the data.")

    if "verified_purchase" in data.columns:
        logger.debug("Running category consistency check for 'verified_purchase'.")
        anomalies_found |= check_category_consistency("verified_purchase", EXPECTED_VERIFIED_PURCHASE_VALUES)
    else:
        logger.debug("Skipping 'verified_purchase' as it is not present in the data.")

    if "categories" in data.columns:
        logger.debug("Running category consistency check for 'categories'.")
        anomalies_found |= check_category_consistency("categories", EXPECTED_CATEGORIES)
    else:
        logger.debug("Skipping 'categories' as it is not present in the data.")

    # 4. Final validation log
    if not anomalies_found:
        logger.info(json.dumps({
            "function": "detect_anomalies",
            "status": "no_anomalies",
            "message": "No anomalies detected in the data."
        }))
    else:
        logger.warning("Anomalies detected during validation.")

    logger.info("Anomaly detection process completed.")
    return not anomalies_found
