# import pandas as pd
# import logging
# import json

# # Review and Title character limits
# REVIEW_LENGTH_MIN = 1
# REVIEW_LENGTH_MAX = 5000
# TITLE_LENGTH_MIN = 1
# TITLE_LENGTH_MAX = 200

# def check_review_title_length(data: pd.DataFrame):
#     """
#     Checks if reviews and titles in the DataFrame meet character length requirements.
    
#     Args:
#         data (pd.DataFrame): DataFrame containing review and title columns.
        
#     Returns:
#         dict: A dictionary with results of short and long reviews and titles.
#     """
#     short_reviews = []
#     long_reviews = []
#     short_titles = []
#     long_titles = []

#     for index, row in data.iterrows():
#         review = row.get('text', "")
#         title = row.get('title', "")
        
#         # Ensure review and title are strings
#         review = str(review) if pd.notnull(review) else ""
#         title = str(title) if pd.notnull(title) else ""
        
#         # Check review length
#         if not (REVIEW_LENGTH_MIN <= len(review) <= REVIEW_LENGTH_MAX):
#             if len(review) < REVIEW_LENGTH_MIN:
#                 short_reviews.append({"index": index, "review": review})
#             elif len(review) > REVIEW_LENGTH_MAX:
#                 long_reviews.append({"index": index, "review": review})

#         # Check title length
#         if not (TITLE_LENGTH_MIN <= len(title) <= TITLE_LENGTH_MAX):
#             if len(title) < TITLE_LENGTH_MIN:
#                 short_titles.append({"index": index, "title": title})
#             elif len(title) > TITLE_LENGTH_MAX:
#                 long_titles.append({"index": index, "title": title})
        
#         # Log any findings
#     if short_reviews or long_reviews:
#         logging.info(json.dumps({
#             "function": "check_review_title_length",
#             "short_reviews": short_reviews,
#             "long_reviews": long_reviews,
#         }))
    
#     if short_titles or long_titles:
#         logging.info(json.dumps({
#             "function": "check_review_title_length",
#             "short_titles": short_titles,
#             "long_titles": long_titles,
#         }))

#     return {
#         "short_reviews": short_reviews,
#         "long_reviews": long_reviews,
#         "short_titles": short_titles,
#         "long_titles": long_titles
#     }


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
        dict: A dictionary with results of short and long reviews and titles.
    """
    # Ensure `text` and `title` columns are strings, fill NaNs with empty strings
    data['text'] = data['text'].fillna("").astype(str)
    data['title'] = data['title'].fillna("").astype(str)
    
    # Check for short or long reviews
    short_reviews = data[data['text'].str.len() < REVIEW_LENGTH_MIN][['text']].to_dict('index')
    long_reviews = data[data['text'].str.len() > REVIEW_LENGTH_MAX][['text']].to_dict('index')

    # Check for short or long titles
    short_titles = data[data['title'].str.len() < TITLE_LENGTH_MIN][['title']].to_dict('index')
    long_titles = data[data['title'].str.len() > TITLE_LENGTH_MAX][['title']].to_dict('index')

    # Log any findings with index information
    if short_reviews or long_reviews:
        logging.info(json.dumps({
            "function": "check_review_title_length",
            "short_reviews": [{"index": idx, "review": info["text"]} for idx, info in short_reviews.items()],
            "long_reviews": [{"index": idx, "review": info["text"]} for idx, info in long_reviews.items()],
        }))
    
    if short_titles or long_titles:
        logging.info(json.dumps({
            "function": "check_review_title_length",
            "short_titles": [{"index": idx, "title": info["title"]} for idx, info in short_titles.items()],
            "long_titles": [{"index": idx, "title": info["title"]} for idx, info in long_titles.items()],
        }))

    return {
        "short_reviews": short_reviews,
        "long_reviews": long_reviews,
        "short_titles": short_titles,
        "long_titles": long_titles
    }


