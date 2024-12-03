import os
import pandas as pd
import gzip
import json
from tqdm import tqdm
from datetime import datetime
import logging
from ..config import CATEGORIES, TARGET_DIRECTORY, SAMPLED_SERVING_DIRECTORY, SERVING_YEAR, SAMPLING_FRACTION
from datetime import datetime, timedelta
import calendar
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def read_jsonl_gz_in_chunks(file_path, chunksize=10000):
    """
    Read a JSONL.GZ file in chunks and yield DataFrames.
    """
    logger.info(f"Reading file: {file_path}")
    with gzip.open(file_path, 'rt') as f:
        while True:
            chunk = []
            for _ in range(chunksize):
                line = f.readline()
                if not line:
                    break
                chunk.append(json.loads(line))
            if not chunk:
                break
            yield pd.DataFrame(chunk)

def load_jsonl_gz(file_path, nrows=None):
    """
    Load a large JSONL.GZ file into a pandas DataFrame.
    """
    chunks = []
    total_rows = 0
    
    for chunk in tqdm(read_jsonl_gz_in_chunks(file_path), desc="Loading data"):
        chunks.append(chunk)
        total_rows += len(chunk)
        if nrows is not None and total_rows >= nrows:
            break
    
    df = pd.concat(chunks, ignore_index=True)
    if nrows is not None:
        df = df.head(nrows)
    
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df

def process_reviews_df(reviews_df, year, month):
    """
    Process the reviews DataFrame based on the provided year and month.
    """
    logger.info(f"Processing reviews DataFrame for {year}-{month:02d}")

    # Ensure month is an integer
    month = int(month)

    # Parse timestamps into datetime and extract required columns
    reviews_df['review_date_timestamp'] = pd.to_datetime(reviews_df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Define start and end dates
    start_date = f"{year}-{month:02d}-01 00:00:00"
    _, last_day = calendar.monthrange(year, month)
    end_date = f"{year}-{month:02d}-{last_day:02d} 23:59:59"
    
    logger.info(f"Filtering reviews from {start_date} to {end_date}")
    
    # Filter DataFrame by date range
    filtered_reviews_df = reviews_df[
        (reviews_df['review_date_timestamp'] >= start_date) & 
        (reviews_df['review_date_timestamp'] <= end_date)
    ].drop(columns=['images'], errors='ignore')
    
    logger.info(f"Filtered reviews DataFrame to {len(filtered_reviews_df)} rows for {start_date} to {end_date}")
    return filtered_reviews_df


def join_dataframes(filtered_reviews_df, metadata_df):
    """
    Join the filtered reviews DataFrame with the metadata DataFrame.
    """
    logger.info("Joining filtered reviews and metadata DataFrames")
    metadata_df_renamed = metadata_df.rename(columns={'title': 'product_name'})
    metadata_df_renamed = metadata_df_renamed[
        ['parent_asin', 'main_category', 'product_name', 'categories', 'price', 'average_rating', 'rating_number']
    ]
    
    joined_df = filtered_reviews_df.merge(
        metadata_df_renamed,
        on='parent_asin',
        how='left'
    )
    
    logger.info(f"Joined DataFrame has {len(joined_df)} rows")
    return joined_df

def sample_data(joined_df):
    """
    Perform stratified sampling on the joined DataFrame.
    """
    logger.info("Performing stratified sampling")
    joined_df['review_month'] = pd.to_datetime(joined_df['review_date_timestamp']).dt.month
    grouped_df = joined_df.groupby(['review_month', 'rating']).size().reset_index(name='count')
    joined_with_count_df = pd.merge(joined_df, grouped_df, on=['review_month', 'rating'], how='inner')
    
    sampled_df = joined_with_count_df.groupby(['review_month', 'rating']).apply(
        lambda x: x.sample(frac=SAMPLING_FRACTION, random_state=42)
    ).reset_index(drop=True)
    
    logger.info(f"Sampled {len(sampled_df)} rows")
    return sampled_df

def save_sampled_data(sampled_df, category_name, year, month):
    """
    Save the sampled data to a CSV file.
    """
    # Ensure `month` is an integer
    month = int(month)

    os.makedirs(SAMPLED_SERVING_DIRECTORY, exist_ok=True)
    file_name = f"sampled_data_{year}_{month:02d}_{category_name}.csv"
    file_path = os.path.join(SAMPLED_SERVING_DIRECTORY, file_name)
    sampled_df.to_csv(file_path, index=False)
    logger.info(f"Saved sampled data to {file_path}")

def sample_serving_data(category_name, month):
    """
    Sample serving data for a specific category within the given year and month.
    """
    try:
        logger.info(f"Processing serving data for category: {category_name}")

        # Convert year and month to integers
        year = SERVING_YEAR
        month = 1

        # File paths
        reviews_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_reviews.jsonl.gz")
        meta_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_meta.jsonl.gz")

        # Load the data
        reviews_df = load_jsonl_gz(reviews_file)
        metadata_df = load_jsonl_gz(meta_file)

        # Process the reviews DataFrame
        filtered_reviews_df = process_reviews_df(reviews_df, year, month)

        # Join DataFrames
        joined_df = join_dataframes(filtered_reviews_df, metadata_df)

        # Sample data
        sampled_df = sample_data(joined_df)

        # Save the sampled data
        save_sampled_data(sampled_df, category_name, year, month)

        logger.info(f"Serving data sampling completed for category: {category_name}")
    except Exception as e:
        logger.error(f"An error occurred while processing serving data for category {category_name}: {e}", exc_info=True)


# def execute_sampling():
#     month = 1  # Start with January
    
#     # Sample data for each category
#     for category in CATEGORIES:
#         sample_serving_data(category, month)

# if __name__ == "__main__":
#     main()
