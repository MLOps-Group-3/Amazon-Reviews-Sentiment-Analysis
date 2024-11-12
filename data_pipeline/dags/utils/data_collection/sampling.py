import sys
import os
import dask.bag as db
import dask.dataframe as dd
import json
import logging
import gzip
from dask.distributed import Client
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Now import from utils
from utils.config import TARGET_DIRECTORY, TARGET_DIRECTORY_SAMPLED, SAMPLING_FRACTION, CATEGORIES

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dask_client():
    """Set up and return a Dask client."""
    logger.info("STEP: Setting up Dask client")
    client = Client("tcp://dask-scheduler:8786")
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client

def preprocess_file(input_file, output_dir, chunk_size=100000):
    """Preprocess the input file and split it into smaller chunks."""
    logger.info(f"STEP: Starting preprocessing of file: {input_file}")
    filename = os.path.basename(input_file)
    output_prefix = os.path.join(output_dir, filename.split('.')[0])
    
    with gzip.open(input_file, 'rt') as f:
        chunk = []
        chunk_num = 0
        total_lines = 0
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                output_file = f"{output_prefix}_{chunk_num}.jsonl.gz"
                with gzip.open(output_file, 'wt') as out:
                    for item in chunk:
                        out.write(json.dumps(item) + '\n')
                logger.info(f"Wrote chunk {chunk_num} to {output_file}")
                chunk = []
                chunk_num += 1
            total_lines += 1
        
        if chunk:
            output_file = f"{output_prefix}_{chunk_num}.jsonl.gz"
            with gzip.open(output_file, 'wt') as out:
                for item in chunk:
                    out.write(json.dumps(item) + '\n')
            logger.info(f"Wrote final chunk {chunk_num} to {output_file}")
    
    logger.info(f"STEP: Finished preprocessing {input_file}. Total lines processed: {total_lines}")

def process_reviews(item):
    """Process a single review item."""
    timestamp = item.get('timestamp', 0)
    review_date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    if '2018-01-01 00:00:00' <= review_date <= '2020-12-31 23:59:59':
        return {
            'asin': item.get('asin'),
            'parent_asin': item.get('parent_asin'),
            'rating': item.get('rating'),
            'timestamp': timestamp,
            'review_date_timestamp': review_date,
            'user_id': item.get('user_id'),  
            'verified_purchase': item.get('verified_purchase'),
            'text': item.get('text'),
            'title': item.get('title'),
            'helpful_vote': item.get('helpful_vote')
        }
    return None

def process_metadata(item):
    """Process a single metadata item."""
    return {
        'parent_asin': item.get('parent_asin'),
        'main_category': item.get('main_category'),
        'product_name': item.get('title', ''),
        'categories': item.get('categories'),
        'price': item.get('price'),
        'average_rating': item.get('average_rating'),
        'rating_number': item.get('rating_number')
    }

def read_jsonl_gz_dask(file_pattern, process_func):
    """Read JSONL.GZ files using Dask and process them in chunks."""
    logger.info(f"STEP: Reading files with pattern: {file_pattern}")
    bag = db.read_text(file_pattern).map(json.loads)
    
    logger.info(f"STEP: Sampling raw data before processing")
    sample_raw = bag.take(5)
    logger.info(f"Sample of raw data: {sample_raw}")
    
    logger.info(f"STEP: Processing data with {process_func.__name__}")
    processed_bag = bag.map(process_func).filter(lambda x: x is not None)
    logger.info(f"STEP: Converting processed data to Dask DataFrame")
    return processed_bag.to_dataframe()

def sample_data(df):
    """Sample data from the DataFrame using Dask DataFrames."""
    logger.info("STEP: Starting data sampling")
    logger.info(f"Initial DataFrame shape: {df.shape[0].compute()}")
    
    logger.info("STEP: Converting timestamp and creating year/month columns")
    df['review_date_timestamp'] = dd.to_datetime(df['review_date_timestamp'])
    df['review_month'] = df['review_date_timestamp'].dt.month
    df['year'] = df['review_date_timestamp'].dt.year
    df['rating'] = dd.to_numeric(df['rating'], errors='coerce')
    
    logger.info("STEP: Grouping data and applying sampling")
    grouped = df.groupby(['review_month', 'rating'])
    sampled = grouped.apply(lambda x: x.sample(frac=SAMPLING_FRACTION, random_state=42), meta=df)
    
    logger.info("STEP: Computing sampled data")
    result = sampled.compute()
    logger.info(f"Sampled data shape: {result.shape}")
    return result

def sample_category(category_name):
    """Process and sample data for a specific category in chunks using Dask DataFrames."""
    client = setup_dask_client()
    try:
        logger.info(f"STEP: Starting data processing for category: {category_name}")
        
        reviews_pattern = os.path.join(TARGET_DIRECTORY_SAMPLED, f"{category_name}_reviews_*.jsonl.gz")
        meta_pattern = os.path.join(TARGET_DIRECTORY_SAMPLED, f"{category_name}_meta_*.jsonl.gz")
        
        logger.info("STEP: Processing reviews")
        reviews_df = read_jsonl_gz_dask(reviews_pattern, process_reviews)
        
        logger.info("STEP: Processing metadata")
        metadata_df = read_jsonl_gz_dask(meta_pattern, process_metadata)
        
        logger.info("STEP: Joining reviews and metadata")
        joined_df = reviews_df.merge(metadata_df, on='parent_asin', how='left')
        
        logger.info("STEP: Sampling data")
        sampled_df = sample_data(joined_df)
        
        logger.info("STEP: Splitting sampled data")
        sampled_2018_2019_df = sampled_df[sampled_df['year'].isin([2018, 2019])]
        sampled_2020_df = sampled_df[sampled_df['year'] == 2020]
        
        logger.info("STEP: Saving sampled data")
        output_2018_2019 = os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2018_2019_{category_name}.csv")
        output_2020 = os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2020_{category_name}.csv")
        
        sampled_2018_2019_df.to_csv(output_2018_2019, index=False)
        sampled_2020_df.to_csv(output_2020, index=False)
        
        logger.info(f"STEP: Data processing completed successfully for {category_name}")
        return True
    except Exception as e:
        logger.exception(f"ERROR: An error occurred during data processing for {category_name}: {str(e)}")
        raise
    finally:
        logger.info("STEP: Closing Dask client")
        client.close()

if __name__ == "__main__":
    logger.info("STEP: Starting sampling process")
    for category in CATEGORIES:
        logger.info(f"STEP: Processing category: {category}")
        sample_category(category)
    logger.info("STEP: Sampling process completed")
