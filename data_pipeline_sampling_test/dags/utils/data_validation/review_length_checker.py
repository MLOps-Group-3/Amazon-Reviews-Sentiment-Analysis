import pandas as pd
import logging

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
        tuple: Individual lists for short and long reviews/titles and status flags.
    """
    data['text'] = data['text'].fillna("").astype(str)
    data['title'] = data['title'].fillna("").astype(str)
    
    # Find indices for short and long reviews and titles
    short_reviews = data[data['text'].str.len() < REVIEW_LENGTH_MIN].index.tolist()
    long_reviews = data[data['text'].str.len() > REVIEW_LENGTH_MAX].index.tolist()
    short_titles = data[data['title'].str.len() < TITLE_LENGTH_MIN].index.tolist()
    long_titles = data[data['title'].str.len() > TITLE_LENGTH_MAX].index.tolist()

    # Set status flags
    short_reviews_flag = not bool(short_reviews)
    long_reviews_flag = not bool(long_reviews)
    short_titles_flag = not bool(short_titles)
    long_titles_flag = not bool(long_titles)

    logging.info("Review and Title length check results", {
        "short_reviews": short_reviews,
        "long_reviews": long_reviews,
        "short_titles": short_titles,
        "long_titles": long_titles,
        "short_reviews_flag": short_reviews_flag,
        "long_reviews_flag": long_reviews_flag,
        "short_titles_flag": short_titles_flag,
        "long_titles_flag": long_titles_flag
    })

    return (
        short_reviews, long_reviews, short_titles, long_titles,
        short_reviews_flag, long_reviews_flag, short_titles_flag, long_titles_flag
    )
