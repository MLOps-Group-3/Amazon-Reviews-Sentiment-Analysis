import sys
import os
import dask.bag as db
import json
import logging
import gzip
from dask.distributed import Client
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Now import from utils
from utils.config import TARGET_DIRECTORY, TARGET_DIRECTORY_SAMPLED, SAMPLING_FRACTION

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dask_client():
    logger.info("Setting up Dask client")
    client = Client("tcp://dask-scheduler:8786")
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client

def preprocess_file(input_file, output_dir, chunk_size=100000):
    logger.info(f"Starting preprocessing of file: {input_file}")
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
    
    logger.info(f"Finished preprocessing {input_file}. Total lines processed: {total_lines}")

def process_reviews(item):
    timestamp = item.get('timestamp', 0)
    review_date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    if '2018-01-01 00:00:00' <= review_date <= '2020-12-31 23:59:59':
        return {
            'asin': item.get('asin'),
            'parent_asin': item.get('parent_asin'),
            'rating': item.get('rating'),
            'review_date_timestamp': review_date,
            'user_id': item.get('user_id'),  
            'verified_purchase': item.get('verified_purchase'),
            'text': item.get('text'),
            'title': item.get('title'),
            'helpful_vote': item.get('helpful_vote')
        }
    return None

def process_metadata(item):
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
    logger.info(f"Reading files with pattern: {file_pattern}")
    bag = db.read_text(file_pattern).map(json.loads)
    
    # Log a few raw items before processing
    sample_raw = bag.take(5)
    logger.info(f"Sample of raw data: {sample_raw}")
    
    processed_bag = bag.map(process_func).filter(lambda x: x is not None)
    logger.info("Computing Dask bag")
    result = processed_bag.compute()
    logger.info(f"Finished computing Dask bag. Number of items: {len(result)}")
    return result


def sample_data(df):
    logger.info("Starting data sampling")
    logger.info(f"Initial DataFrame shape: {df.shape}")
    logger.info(f"Initial DataFrame columns: {df.columns}")
    
    df['review_date_timestamp'] = pd.to_datetime(df['review_date_timestamp'])
    df['review_month'] = df['review_date_timestamp'].dt.month
    df['year'] = df['review_date_timestamp'].dt.year
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    logger.info(f"DataFrame columns after date processing: {df.columns}")
    
    grouped = df.groupby(['review_month', 'rating'])
    group_sizes = grouped.size().reset_index(name='count')
    df = df.merge(group_sizes, on=['review_month', 'rating'])
    
    logger.info(f"DataFrame columns after grouping: {df.columns}")
    
    sampled = df.groupby(['review_month', 'rating']).apply(
        lambda x: x.sample(frac=SAMPLING_FRACTION, random_state=42)
    ).reset_index(drop=True)
    
    logger.info(f"DataFrame columns after sampling: {sampled.columns}")
    
    sampled = sampled.drop(columns=['count'])
    logger.info(f"Sampled data shape: {sampled.shape}")
    logger.info(f"Final sampled DataFrame columns: {sampled.columns}")
    return sampled

def sample_category(category_name):
    client = setup_dask_client()
    try:
        logger.info(f"Starting data processing for category: {category_name}")
        
        reviews_pattern = os.path.join(TARGET_DIRECTORY_SAMPLED, f"{category_name}_reviews_*.jsonl.gz")
        meta_pattern = os.path.join(TARGET_DIRECTORY_SAMPLED, f"{category_name}_meta_*.jsonl.gz")
        
        logger.info(f"Processing reviews for {category_name}")
        reviews_df = pd.DataFrame(read_jsonl_gz_dask(reviews_pattern, process_reviews))
        logger.info(f"Reviews DataFrame shape: {reviews_df.shape}")
        logger.info(f"Reviews DataFrame columns: {reviews_df.columns}")
        
        logger.info(f"Processing metadata for {category_name}")
        metadata_df = pd.DataFrame(read_jsonl_gz_dask(meta_pattern, process_metadata))
        logger.info(f"Metadata DataFrame shape: {metadata_df.shape}")
        logger.info(f"Metadata DataFrame columns: {metadata_df.columns}")
        
        logger.info(f"Joining reviews and metadata for {category_name}")
        joined_df = reviews_df.merge(metadata_df, on='parent_asin', how='left')
        logger.info(f"Joined DataFrame shape: {joined_df.shape}")
        logger.info(f"Joined DataFrame columns: {joined_df.columns}")
        
        joined_df['categories'] = joined_df['categories'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
        sampled_df = sample_data(joined_df)
        
        logger.info(f"Splitting sampled data for {category_name}")
        sampled_2018_2019_df = sampled_df[sampled_df['year'].isin([2018, 2019])]
        sampled_2020_df = sampled_df[sampled_df['year'] == 2020]
        
        logger.info(f"2018-2019 DataFrame columns: {sampled_2018_2019_df.columns}")
        logger.info(f"2020 DataFrame columns: {sampled_2020_df.columns}")
        
        output_2018_2019 = os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2018_2019_{category_name}.csv")
        output_2020 = os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2020_{category_name}.csv")
        
        logger.info(f"Saving sampled data for {category_name}")
        sampled_2018_2019_df.to_csv(output_2018_2019, index=False)
        sampled_2020_df.to_csv(output_2020, index=False)
        
        logger.info(f"Data processing completed successfully for {category_name}")
        return True
    except Exception as e:
        logger.exception(f"An error occurred during data processing for {category_name}: {str(e)}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    logger.info("Starting sampling process")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    for category in CATEGORIES:
        logger.info(f"Processing category: {category}")
        sample_category(category)
    logger.info("Sampling process completed")
