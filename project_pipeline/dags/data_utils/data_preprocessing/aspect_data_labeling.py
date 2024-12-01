import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define sentiment labels
POSITIVE = 1
NEGATIVE = 2
NEUTRAL = 0

# Label mapping
label_mapping = {
    POSITIVE: "POSITIVE",
    NEGATIVE: "NEGATIVE",
    NEUTRAL: "NEUTRAL"
}

def vader_sentiment_label(row):
    """
    Uses VADER sentiment analyzer to classify the sentiment of review text.

    Returns:
      - POSITIVE if VADER's compound sentiment score is 0.05 or higher.
      - NEGATIVE if the score is -0.05 or lower.
      - NEUTRAL if the score falls between -0.05 and 0.05.

    Parameters:
        row (pd.Series): A Series representing a row in the DataFrame, with 'relevant_sentences' containing review text.

    Returns:
        int: Label (POSITIVE, NEGATIVE, or NEUTRAL).
    """
    if row["relevant_sentences"]:
        score = vader_analyzer.polarity_scores(row["relevant_sentences"])["compound"]
        if score >= 0.05:
            return POSITIVE
        elif score <= -0.05:
            return NEGATIVE
    return NEUTRAL

def apply_vader_labeling(df_cleaned):
    """
    Applies VADER sentiment analysis to classify each review's sentiment and maps numeric labels to descriptive strings.

    Parameters:
        df_cleaned (DataFrame): Cleaned pandas DataFrame containing reviews.

    Returns:
        DataFrame: DataFrame with predicted labels added as a new column.
    """
    logging.info("Applying VADER sentiment analysis...")
    start_time = time.time()
    
    # Apply VADER sentiment labeling to each row
    df_cleaned['sentiment_label'] = df_cleaned.apply(vader_sentiment_label, axis=1)
    apply_time = time.time() - start_time
    logging.info(f"VADER sentiment analysis applied in {apply_time:.2f} seconds.")
    
    # Map numeric labels to descriptive labels
    df_cleaned['sentiment_label'] = df_cleaned['sentiment_label'].map(label_mapping)
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage with file input
    file_path = "milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/aspect_based.csv"
    
    # Load data
    df_raw = pd.read_csv(file_path)
    logging.info(f"Data loaded from {file_path}.")
    
    # Ensure necessary columns are present
    if 'relevant_sentences' not in df_raw.columns:
        logging.error("The 'relevant_sentences' column is missing from the DataFrame.")
    else:
        # Apply VADER sentiment analysis
        labeled_data = apply_vader_labeling(df_raw)
        
        # Save the labeled data
        output_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/labeled_data/labeled_data_vader.csv'
        labeled_data.to_csv(output_path, index=False)
        logging.info(f"Labeled data saved to '{output_path}'.")

        # Display labeled data distribution
        print(labeled_data['sentiment_label'].value_counts())
