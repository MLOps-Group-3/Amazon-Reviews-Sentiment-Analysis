import pandas as pd
import gzip
import json
from tqdm import tqdm
import logging
import logging.config
import os
from datetime import datetime
from ..config import CATEGORIES, TARGET_DIRECTORY_SAMPLED, TARGET_DIRECTORY, SAMPLING_FRACTION

# Set up logging configuration
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file_handler': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            'mode': 'w'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file_handler'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
})

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

def process_reviews_df(reviews_df):
    """
    Process the reviews DataFrame.
    """
    logger.info("Processing reviews DataFrame")
    reviews_df['review_date_timestamp'] = pd.to_datetime(reviews_df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    filtered_reviews_df = reviews_df[
        (reviews_df['review_date_timestamp'] >= '2022-01-01 00:00:00') & 
        (reviews_df['review_date_timestamp'] <= '2023-12-31 23:59:59')
    ].drop(columns=['images'])
    
    logger.info(f"Filtered reviews DataFrame to {len(filtered_reviews_df)} rows")
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

def process_sampled_data(sampled_df):
    """
    Process the sampled data and split into two DataFrames.
    """
    logger.info("Processing sampled data")
    sampled_df['review_date_timestamp'] = pd.to_datetime(sampled_df['review_date_timestamp'], format='%Y-%m-%d %H:%M:%S')
    sampled_df['year'] = sampled_df['review_date_timestamp'].dt.year
    sampled_df['categories'] = sampled_df['categories'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    sampled_df = sampled_df.drop(columns=['count'])
    
    sampled_2018_2019_df = sampled_df[sampled_df['year'].isin([2022, 2023])]
    sampled_2020_df = sampled_df[sampled_df['year'] == 2021]
    
    logger.info(f"Split data into {len(sampled_2018_2019_df)} rows for 2018-2019 and {len(sampled_2020_df)} rows for 2020")
    return sampled_2018_2019_df, sampled_2020_df

def sample_category(category_name):
    try:
        logger.info(f"Starting data processing for category: {category_name}")
        
        # File paths
        reviews_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_reviews.jsonl.gz")
        meta_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_meta.jsonl.gz")
        
        # Load the data
        reviews_df = load_jsonl_gz(reviews_file)
        metadata_df = load_jsonl_gz(meta_file)
        
        # Process the reviews DataFrame
        filtered_reviews_df = process_reviews_df(reviews_df)
        
        # Join DataFrames
        joined_df = join_dataframes(filtered_reviews_df, metadata_df)
        
        # Sample data
        sampled_df = sample_data(joined_df)
        
        # Process sampled data
        sampled_2018_2019_df, sampled_2020_df = process_sampled_data(sampled_df)
        
        # Save to CSV
        os.makedirs(TARGET_DIRECTORY_SAMPLED, exist_ok=True)
        sampled_2018_2019_df.to_csv(os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2022_2023_{category_name}.csv"), index=False)
        sampled_2020_df.to_csv(os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2021_{category_name}.csv"), index=False)
        
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during data processing: {str(e)}")

