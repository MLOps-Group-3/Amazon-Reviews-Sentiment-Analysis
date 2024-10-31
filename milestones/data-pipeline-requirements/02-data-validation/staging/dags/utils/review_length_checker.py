import pandas as pd
import logging
import json

# Review and Title character limits
REVIEW_LENGTH_MIN = 1
REVIEW_LENGTH_MAX = 5000
TITLE_LENGTH_MIN = 1
TITLE_LENGTH_MAX = 200

def check_review_title_length(data: pd.DataFrame):
    """
    Checks if reviews and titles in the DataFrame meet character length requirements.
    
    Args:
        data (pd.DataFrame): DataFrame containing review and title columns.
        
    Returns:
        dict: A dictionary with lists of row indices for short and long reviews and titles,
              and flags indicating if any issues were found.
    """
    # Ensure `text` and `title` columns are strings, fill NaNs with empty strings
    data['text'] = data['text'].fillna("").astype(str)
    data['title'] = data['title'].fillna("").astype(str)
    
    # Find indices for short and long reviews
    short_reviews = data[data['text'].str.len() < REVIEW_LENGTH_MIN].index.tolist()
    long_reviews = data[data['text'].str.len() > REVIEW_LENGTH_MAX].index.tolist()

    # Find indices for short and long titles
    short_titles = data[data['title'].str.len() < TITLE_LENGTH_MIN].index.tolist()
    long_titles = data[data['title'].str.len() > TITLE_LENGTH_MAX].index.tolist()

    # Determine flags for the presence of short or long reviews/titles
    status_flags = {
        "short_reviews_flag": bool(short_reviews),
        "long_reviews_flag": bool(long_reviews),
        "short_titles_flag": bool(short_titles),
        "long_titles_flag": bool(long_titles)
    }

    # Log results with row indices and status flags
    logging.info(json.dumps({
        "function": "check_review_title_length",
        "short_reviews": short_reviews,
        "long_reviews": long_reviews,
        "short_titles": short_titles,
        "long_titles": long_titles,
        "status_flags": status_flags
    }))

    return {
        "short_reviews": short_reviews,
        "long_reviews": long_reviews,
        "short_titles": short_titles,
        "long_titles": long_titles,
        "status_flags": status_flags
    }

# Example usage:
# df = pd.read_csv("path_to_your_data.csv")
# length_issues = check_review_title_length(df)
