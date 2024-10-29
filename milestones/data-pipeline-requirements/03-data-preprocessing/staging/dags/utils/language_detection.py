"""
File : utils/language_detection.py
"""

import pandas as pd
import logging
from langdetect import detect, DetectorFactory, LangDetectException
from concurrent.futures import ThreadPoolExecutor

# Ensure consistent results from the language detection library
DetectorFactory.seed = 0

def detect_comment_language(text, idx):
    """
    Detect the language of a given comment.

    Parameters:
    text : str 
        The comment to detect language for.
    idx : int 
        The row number of the comment.

    Returns:
    tuple: (row number, detected language or None in case of error)
    """
    try:
        # Check if the text is valid
        if not isinstance(text, str) or not text.strip():
            logging.warning(f"Row {idx}: Skipping empty or non-meaningful text.")
            return idx, None

        detected_lang = detect(text)
        return idx, detected_lang
    except LangDetectException as e:
        logging.error(f"Row {idx}: Error detecting language. Exception: {str(e)}")
        return idx, None

def detect_language(data: pd.DataFrame, text_column: str = "text") -> (list, bool):
    """
    Detect the language of comments in a specified text column. If any comment is non-English,
    it will be logged along with the row number, and the function will return a list of 
    non-English row numbers and a success flag.

    Parameters:
    data : DataFrame 
        The input data containing a column with text to be analyzed.
    text_column : str 
        The name of the column with text to detect language. Default is 'text'.

    Returns:
    tuple: (list of non-English row numbers, bool indicating success
    """
    non_english_rows = []

    if text_column not in data.columns:
        logging.error(f"Column '{text_column}' not found in the data.")
        return non_english_rows, False

    logging.info(f"Starting language detection on column: '{text_column}'.")

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda x: detect_comment_language(x[1], x[0]),
            enumerate(data[text_column])
        ))

    # Process the results and log non-English comments
    for idx, lang in results:
        if lang and lang != "en":
            logging.warning(f"Row {idx}: Non-English comment detected (Language: {lang}).")
            non_english_rows.append(idx)

    if non_english_rows:
        logging.info(f"Language detection completed. Non-English rows: {non_english_rows}.")
        return non_english_rows, False
    else:
        logging.info("All comments are in English.")
        return non_english_rows, True

# Example usage:
if __name__ == "__main__":
    data = pd.DataFrame({
        'text': ['Hello world!', '', 'Bonjour le monde!', None, '    ', 'Hola mundo!']
    })
    result = detect_language(data)
    print(result)
