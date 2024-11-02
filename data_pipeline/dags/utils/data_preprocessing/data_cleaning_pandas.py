import pandas as pd
import re
import emoji
import logging
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_amazon_reviews(df_raw, emoji_indices):
    """
    Function to clean Amazon review data using Pandas.
    
    Parameters:
        df_raw (DataFrame): Raw DataFrame containing Amazon review data.
        emoji_indices (list): List of row indices where emojis need to be converted to text.
    
    Returns:
        DataFrame: Cleaned Pandas DataFrame.
    """
    
    logging.info("Starting data cleaning process")

    # 1. Drop duplicates
    original_shape = df_raw.shape
    df_raw = df_raw.drop_duplicates()
    logging.info(f"Dropped duplicates: {original_shape[0] - df_raw.shape[0]} rows removed")

    # 2. Drop reviews with missing text or rating
    df_raw = df_raw.dropna(subset=["text", "rating"])
    logging.info(f"Dropped rows with missing text or rating: {original_shape[0] - df_raw.shape[0]} rows removed")

    # 3. Convert dtype based on range to reduce memory usage
    df_raw['rating'] = df_raw['rating'].astype('int8')
    df_raw["helpful_vote"] = df_raw["helpful_vote"].astype('int32')
    df_raw["average_rating"] = df_raw["average_rating"].astype('float32')
    df_raw["year"] = df_raw["year"].astype('int16')
    df_raw["review_month"] = df_raw["review_month"].astype('int8')
    logging.info("Converted data types to reduce memory usage")

    # 4. Handle missing values
    df_raw["title"].fillna("unknown", inplace=True)
    df_raw["helpful_vote"].fillna(0, inplace=True)
    df_raw["price"] = df_raw["price"].apply(lambda x: "unknown" if pd.isnull(x) or x == "NULL" else str(x))
    logging.info("Handled missing values in title, helpful_vote, and price columns")

    # 5. Clean and normalize text columns
    df_raw["title"] = df_raw["title"].apply(lambda x: re.sub(r"\s+", " ", x))
    df_raw["text"] = df_raw["text"].apply(lambda x: re.sub(r"\s+", " ", x))
    df_raw["text"] = df_raw["text"].apply(lambda x: re.sub(r"<.*?>", " ", x))
    df_raw["title"] = df_raw["title"].apply(lambda x: re.sub(r"<.*?>", " ", x))
    logging.info("Removed extra whitespaces and HTML tags from text and title columns")

    # 6. Apply emoji demojization only to valid indices
    valid_emoji_indices = list(set(emoji_indices) & set(df_raw.index))
    df_raw.loc[valid_emoji_indices, "text"] = df_raw.loc[valid_emoji_indices, "text"].apply(emoji.demojize)
    logging.info(f"Demojized text for {len(valid_emoji_indices)} rows")
    logging.info("Data cleaning process completed")
    
    return df_raw

if __name__ == "__main__":
    # Example usage
    file_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/sampled_data_2018_2019.csv'
    df_raw = pd.read_csv(file_path)
    logging.info(f"Loaded raw data from {file_path} with shape {df_raw.shape}")

    # Load validation data to get emoji indices
    validation_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/validation_data/validation_results.csv'
    validation_data = pd.read_csv(validation_path)
    emoji_indices = eval(validation_data.loc[validation_data["function"] == "emoji_detection", "row_indices"].values[0])

    # Clean the data
    df_cleaned = clean_amazon_reviews(df_raw, emoji_indices)
    logging.info(f"Cleaned data shape: {df_cleaned.shape}")

    # Display the count of rows where price is "unknown" after cleaning
    unknown_price_count = (df_cleaned["price"] == "unknown").sum()
    logging.info(f"Rows with 'unknown' price after cleaning: {unknown_price_count}")

    # Save the cleaned data
    cleaned_file_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/cleaned_data_2018_2019.csv'
    df_cleaned.to_csv(cleaned_file_path, index=False)
    logging.info(f"Cleaned data saved to {cleaned_file_path}")
