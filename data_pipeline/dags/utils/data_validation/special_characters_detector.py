import pandas as pd
import logging
import re

# Define a pattern for reviews that contain only special characters
ONLY_SPECIAL_CHAR_PATTERN = r'^[!@#$%^&*()_+\-=\[\]{};:\\|,.<>\/?]+$'

def check_only_special_characters(data: pd.DataFrame):
    """
    Identifies reviews that consist solely of special characters and logs the row indexes along with the review text.
    
    Args:
        data (pd.DataFrame): DataFrame containing review texts.
        
    Returns:
        list: A list of indexes where reviews contain only special characters.
    """
    invalid_reviews = []

    # Function to check for only special characters
    for index, review in enumerate(data['text']):
        if isinstance(review, str) and re.match(ONLY_SPECIAL_CHAR_PATTERN, review):
            invalid_reviews.append(index)
            logging.error(f"Row {index}: Review contains only special characters: '{review}'")

    # Log results
    if invalid_reviews:
        logging.error(f"Reviews containing only special characters found at indexes: {invalid_reviews}")
    else:
        logging.info("No reviews found that contain only special characters.")

    # Log a warning if there are no reviews at all
    if data['text'].isnull().all():
        logging.warning("All reviews are missing (null). Please check the data.")

    return invalid_reviews

# Example usage:
# df = pd.read_csv("path_to_your_data.csv")
# invalid_indices = check_only_special_characters(df)
