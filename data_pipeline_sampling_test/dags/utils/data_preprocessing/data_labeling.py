import pandas as pd
from snorkel.labeling import labeling_function, LFApplier
from snorkel.labeling.model import LabelModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time 
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Define sentiment labels as strings
POSITIVE = 1
NEGATIVE = 2
NEUTRAL = 0

# Define keywords for positive and negative sentiment
positive_words = ["excellent", "amazing", "great", "fantastic", "love", "wonderful", "perfect", "satisfied", "good", "best", "recommend"]
negative_words = ["bad", "poor", "terrible", "awful", "disappointed", "waste", "horrible", "worst", "not worth", "refund", "broken", "defective"]

@labeling_function()
def lf_positive_negative_ratio(row):
    """
    Labeling function that calculates the ratio of positive to negative words in review text.

    If both positive and negative words are present:
      - Returns "Positive" if the ratio of positive words to negative words is greater than 1.1.
      - Returns "Negative" if the ratio is less than 0.9.
      - Returns "Neutral" if the ratio falls between 0.9 and 1.1.
    If only positive or only negative words are present:
      - Returns "Positive" or "Negative", respectively.
    If neither is present, returns "Neutral".

    Parameters:
        row (dict): A dictionary representing a row in the DataFrame, with 'text' containing review text.

    Returns:
        str: Label ("Positive", "Negative", or "Neutral").
    """
    if row["text"]:
        positive_count = sum(row["text"].lower().count(word) for word in positive_words)
        negative_count = sum(row["text"].lower().count(word) for word in negative_words)
        
        if positive_count > 0 and negative_count > 0:
            ratio = positive_count / negative_count
            if 0.9 <= ratio <= 1.1:
                return NEUTRAL
            elif ratio > 1.1:
                return POSITIVE
            else:
                return NEGATIVE
        elif positive_count > 0:
            return POSITIVE
        elif negative_count > 0:
            return NEGATIVE
    return NEUTRAL

@labeling_function()
def lf_high_rating(row):
    """
    Labeling function that labels reviews with a high rating as "Positive".

    Returns "Positive" if the 'rating' column is 4 or higher.
    Returns "Neutral" otherwise.

    Parameters:
        row (dict): A dictionary representing a row in the DataFrame, with 'rating' containing review rating.

    Returns:
        str: Label ("Positive" or "Neutral").
    """
    return POSITIVE if row["rating"] >= 4 else NEUTRAL

@labeling_function()
def lf_low_rating(row):
    """
    Labeling function that labels reviews with a low rating as "Negative".

    Returns "Negative" if the 'rating' column is 2 or lower.
    Returns "Neutral" otherwise.

    Parameters:
        row (dict): A dictionary representing a row in the DataFrame, with 'rating' containing review rating.

    Returns:
        str: Label ("Negative" or "Neutral").
    """
    return NEGATIVE if row["rating"] <= 2 else NEUTRAL

@labeling_function()
def lf_vader_sentiment(row):
    """
    Labeling function that uses the VADER sentiment analyzer to classify text.

    Returns:
      - "Positive" if VADER's compound sentiment score is 0.05 or higher.
      - "Negative" if the score is -0.05 or lower.
      - "Neutral" if the score falls between -0.05 and 0.05.

    Parameters:
        row (dict): A dictionary representing a row in the DataFrame, with 'text' containing review text.

    Returns:
        str: Label ("Positive", "Negative", or "Neutral").
    """
    if row["text"]:
        score = vader_analyzer.polarity_scores(row["text"])["compound"]
        if score >= 0.05:
            return POSITIVE
        elif score <= -0.05:
            return NEGATIVE
    return NEUTRAL

def apply_labelling(df_cleaned):
    """
    Function to apply labeling functions and train a label model.

    Parameters:
        df_cleaned (DataFrame): Cleaned pandas DataFrame containing reviews.

    Returns:
        DataFrame: DataFrame with predicted labels added as a new column.
    """
    # lfs = [lf_positive_negative_ratio, lf_high_rating, lf_low_rating, lf_vader_sentiment]
    try:
        # Convert DataFrame to list of dictionaries
        data_points = df_cleaned.to_dict(orient="records")
        lfs = [lf_high_rating, lf_low_rating, lf_vader_sentiment]
        logging.info("Applying labeling functions...")
        
        # Apply labeling functions
        try:
            applier = LFApplier(lfs)
            start_time = time.time()
            L = applier.apply(data_points)
            apply_time = time.time() - start_time
            logging.info(f"Labeling functions applied in {apply_time:.2f} seconds.")
        except Exception as e:
            logging.error("Error applying labeling functions.", exc_info=True)
            raise e

        # Train LabelModel
        try:
            label_model = LabelModel(cardinality=3, verbose=True)
            logging.info("Fitting the label model...")
            start_time = time.time()
            label_model.fit(L_train=L, n_epochs=500, log_freq=100)
            fit_time = time.time() - start_time
            logging.info(f"Model fitting completed in {fit_time:.2f} seconds.")
        except Exception as e:
            logging.error("Error fitting the label model.", exc_info=True)
            raise e

        # Predict labels
        try:
            label_mapping = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}
            logging.info("Predicting labels...")
            start_time = time.time()
            df_cleaned['sentiment_label'] = [label_mapping[label] for label in label_model.predict(L)]
            predict_time = time.time() - start_time
            logging.info(f"Prediction on all rows completed in {predict_time:.2f} seconds.")
        except Exception as e:
            logging.error("Error during prediction.", exc_info=True)
            raise e

    except Exception as e:
        logging.critical("An error occurred in the labeling and prediction process.", exc_info=True)
        raise e

    return df_cleaned

if __name__ == "__main__":
    # Example usage with file input
    file_path = "milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/cleaned_data_2018_2019.csv"
    df_raw = pd.read_csv(file_path)
    print(df_raw.dtypes)
    # Ensure necessary columns are numeric as expected
    df_raw['rating'] = pd.to_numeric(df_raw['rating'], errors='coerce')
    
    # Apply labeling function
    labeled_data = apply_labelling(df_raw)
    
    labeled_data.to_csv('milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/labeled_data/labeled_data_2018_2019.csv')
    # Display labeled data distribution
    print(labeled_data['sentiment_label'].value_counts())
