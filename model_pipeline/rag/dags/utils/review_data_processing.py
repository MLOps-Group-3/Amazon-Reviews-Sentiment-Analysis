import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """
    Loads and processes the data from the specified CSV file.
    
    Args:
        file_path (str): Path to the CSV file to load.
        
    Returns:
        pd.DataFrame: Processed DataFrame with columns 'review_month' and 'year' added.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.drop(columns=['user_id', 'asin'])  # Drop unnecessary columns
        df['review_date_timestamp'] = pd.to_datetime(df['review_date_timestamp'])  # Convert to datetime
        df['review_month'] = df['review_date_timestamp'].dt.month  # Extract month
        df['year'] = df['review_date_timestamp'].dt.year  # Extract year
        logger.info(f"Data loaded and processed successfully from {file_path}.")
        return df
    except Exception as e:
        logger.error(f"Error loading or processing data: {e}")
        raise ValueError(f"Error loading or processing data: {e}")

def extract_third_or_last_category(category: str, main_category: str) -> str:
    """
    Extracts the third or last level of a category string, falling back to the main category if necessary.
    
    Args:
        category (str): The category string to extract from.
        main_category (str): The fallback main category if the 'category' is not suitable.
        
    Returns:
        str: Extracted category (third or last level).
    """
    if pd.isna(category) or not category.strip():  # If category is missing or empty
        category = main_category  # Use the main category as fallback
    levels = category.split(',')
    return levels[2] if len(levels) >= 3 else levels[-1]  # Return the third level or last level

def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates data by year, review_month, and extracted category, including sentiment and helpful votes.
    
    Args:
        df (pd.DataFrame): The DataFrame to aggregate.
        
    Returns:
        pd.DataFrame: Aggregated data by year, review_month, and extracted_category.
    """
    try:
        # Extract the category or fallback to the main category
        df['extracted_category'] = df.apply(
            lambda row: extract_third_or_last_category(row['categories'], row['main_category']), axis=1
        )

        # Aggregate data
        aggregated = df.groupby(
            ['year', 'review_month', 'extracted_category']
        ).agg(
            text=('text', lambda x: ' '.join(x)),  # Combine text from reviews
            average_rating=('rating', 'mean'),  # Mean rating
            avg_helpful_votes=('helpful_vote', 'mean'),  # Mean helpful votes
            sentiment=('sentiment_label', lambda x: x.mode()[0])  # Most common sentiment
        ).reset_index()

        logger.info(f"Data aggregated successfully into monthly groups.")
        return aggregated
    except Exception as e:
        logger.error(f"Error during data aggregation: {e}")
        raise ValueError(f"Error during data aggregation: {e}")

def prepare_documents(df: pd.DataFrame) -> List[Dict]:
    """
    Convert the DataFrame into a list of documents to be used by the RAG model.
    
    Args:
        df (pd.DataFrame): Aggregated reviews DataFrame.
        
    Returns:
        List[Dict]: List of documents, each being a dictionary.
    """
    documents = []

    for _, row in df.iterrows():
        document = {
            'text': row['text'],
            'year': row['year'],
            'month': row['review_month'],
            'category': row['extracted_category'],
            'average_rating': row['average_rating'],
            'avg_helpful_votes': row['avg_helpful_votes'],
            'dominant_sentiment': row['sentiment'],  # Sentiment column
        }
        documents.append(document)

    logger.info(f"Prepared {len(documents)} documents for RAG model.")
    return documents

def save_documents(documents: List[Dict], file_path: str) -> None:
    """
    Save the documents in JSON format for later retrieval by the RAG model.
    
    Args:
        documents (List[Dict]): List of documents.
        file_path (str): Path where the documents should be saved.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(documents, f, indent=4)
        logger.info(f"Documents saved to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving documents to {file_path}: {e}")
        raise

def main(input_csv: str, output_json: str) -> None:
    """
    Main function to load data, prepare documents, and save them in JSON format.
    
    Args:
        input_csv (str): Path to the aggregated CSV data.
        output_json (str): Path where the processed documents will be saved.
    """
    # Load and process the data
    df = load_and_process_data(input_csv)

    # Ensure necessary columns are present
    required_columns = ['text', 'year', 'review_month', 'categories', 'main_category', 'rating', 'helpful_vote', 'sentiment_label']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns in input CSV: {required_columns}")
        raise ValueError(f"Missing required columns in input CSV: {required_columns}")

    # Aggregate the data
    aggregated_df = aggregate_data(df)

    # Prepare documents for RAG model
    documents = prepare_documents(aggregated_df)

    # Save documents to a JSON file
    save_documents(documents, output_json)

if __name__ == '__main__':
    # Define file paths for input and output
    input_csv = '/Users/praneethkorukonda/Downloads/Amazon-Reviews-Sentiment-Analysis/data_pipeline/data/labeled/labeled_data.csv'  # Input file path (aggregated data)
    output_json = '/Users/praneethkorukonda/Downloads/Amazon-Reviews-Sentiment-Analysis/data_pipeline/data/document_store.json'  # Output file path (JSON for RAG model)

    # Run the main function
    main(input_csv, output_json)
