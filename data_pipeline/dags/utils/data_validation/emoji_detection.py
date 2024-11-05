"""
File : utils/emoji_detection.py
"""

import pandas as pd
import logging
import re

# Regular expression pattern for detecting emojis
EMOJI_PATTERN = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F700-\U0001F77F"  # alchemical symbols
    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    u"\U00002702-\U000027B0"  # Dingbats
    u"\U000024C2-\U0001F251"  # Enclosed characters
    "]+", flags=re.UNICODE
)

def detect_emoji(data: pd.DataFrame, text_column: str = "text") -> (list, bool):
    """
    Detect emojis in a specified text column. If any emojis are found, log them along with the row number,
    and the function will return a list of row numbers containing emojis and a success flag.

    Parameters:
    data : DataFrame 
        The input data containing a column with text to be analyzed.
    text_column : str 
        The name of the column with text to detect emojis. Default is 'text'.

    Returns:
    tuple: (list of row numbers containing emojis, bool indicating success)
    """
    emoji_rows = []

    if text_column not in data.columns:
        logging.error(f"Column '{text_column}' not found in the data.")
        return emoji_rows, False

    logging.info(f"Starting emoji detection on column: '{text_column}'.")

    # Iterate through the text column and detect emojis
    for idx, text in enumerate(data[text_column]):
        if EMOJI_PATTERN.search(str(text)):
            logging.warning(f"Row {idx}: Emoji detected in text: '{text}'")
            emoji_rows.append(idx)

    if emoji_rows:
        logging.info(f"Emoji detection completed. Rows with emojis: {emoji_rows}.")
        return emoji_rows, False
    else:
        logging.info("No emojis found in the text.")
        return emoji_rows, True

# Example usage:
if __name__ == "__main__":
    data = pd.read_csv("/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/milestones/data-pipeline-requirements/02-data-validation/staging/data/sampled_data_2018_2019.csv")
    result = detect_emojis(data)
    print(result)
