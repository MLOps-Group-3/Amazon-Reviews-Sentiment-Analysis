import pandas as pd
import re
import json
import logging
import os

# Create a log directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

def detect_non_english(review):
    """
    Detects if a review is non-English and identifies its language.
    
    Args:
        review (str): The review text to analyze.
        
    Returns:
        tuple: A tuple containing the detected language and a boolean indicating if it's non-English.
    """
    # This is a placeholder for actual language detection logic
    # Using a simple regex for demonstration (adjust as needed for actual language detection)
    if re.search(r'[^\x00-\x7F]', review):  # Non-ASCII characters indicate non-English
        language = 'es'  # Just for demonstration, set to Spanish
        is_non_english = True
    else:
        language = 'en'
        is_non_english = False
    
    # Log details only for non-English reviews
    if is_non_english:
        logging.info({"function": "detect_non_english", "language": language, "is_non_english": is_non_english, "review": review})
    
    return language, is_non_english

def detect_emojis(review):
    """
    Detects emojis in a review.
    
    Args:
        review (str): The review text to analyze.
        
    Returns:
        list: A list of detected emojis.
    """
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"
                                "\U0001F300-\U0001F5FF"
                                "\U0001F680-\U0001F6FF"
                                "\U0001F700-\U0001F77F"
                                "\U0001F780-\U0001F7FF"
                                "\U0001F800-\U0001F8FF"
                                "\U0001F900-\U0001F9FF"
                                "\U0001FA00-\U0001FAFF"
                                "\U00002702-\U000027B0"
                                "\u2600-\u26FF"
                                "\u2700-\u27BF]", flags=re.UNICODE)
    
    emojis = emoji_pattern.findall(review)
    
    # Log details only if emojis are detected
    if emojis:
        logging.info({"function": "detect_emojis", "emojis": emojis, "review": review})
    
    return emojis

def process_reviews(df):
    """
    Processes the DataFrame to detect non-English reviews and emojis.
    
    Args:
        df (pd.DataFrame): DataFrame containing review texts.
        
    Returns:
        dict: A dictionary with keys 'non_english_reviews' and 'emoji_reviews'.
    """
    non_english_reviews = []
    emoji_reviews = []

    for index, row in df.iterrows():
        review = row['text']
        
        # Check if review is a string
        if isinstance(review, str):
            # Detect non-English reviews
            language, is_non_english = detect_non_english(review)
            if is_non_english:
                non_english_reviews.append(review)

            # Detect emojis
            emojis = detect_emojis(review)
            if emojis:
                emoji_reviews.append({'review': review, 'emojis': emojis})
        else:
            logging.warning({"function": "process_reviews", "message": "Non-string value found", "index": index, "value": str(review)})

    return {
        'non_english_reviews': non_english_reviews,
        'emoji_reviews': emoji_reviews
    }

# Example usage (replace this with your DataFrame)
if __name__ == "__main__":
    # Example DataFrame
    data_path = "/Users/vallimeenaa/Desktop/Group 3 Project/Amazon-Reviews-Sentiment-Analysis/data/sampled_2020_df.csv"

    # Load data from CSV file
    data = pd.read_csv(data_path)

    # Run the checkers and log results
    result_1 = process_reviews(data)

    # Log the results
    logger.info(json.dumps({
        "function": "process_reviews",
        "result": result_1,
        "message": "Review processing completed."
    }))