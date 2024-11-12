import sys
import os
import dask.bag as db
import dask.dataframe as dd
import json
import logging
import gcsfs
from dask.distributed import Client
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from utils.config import GCS_RAW_DATA_PATH, GCS_SAMPLED_DATA_PATH, SAMPLING_FRACTION, CATEGORIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dask_client():
    logger.info("STEP: Setting up Dask client")
    client = Client("tcp://dask-scheduler:8786")
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client

def preprocess_file(input_file, output_dir, chunk_size=100000):
    logger.info(f"STEP: Starting preprocessing of file: {input_file}")
    fs = gcsfs.GCSFileSystem()
    filename = os.path.basename(input_file)
    output_prefix = f"{output_dir}/{filename.split('.')[0]}"

    with fs.open(input_file, 'rt') as f:
        chunk = []
        chunk_num = 0
        total_lines = 0
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                output_file = f"{output_prefix}_{chunk_num}.jsonl.gz"
                with fs.open(output_file, 'wt') as out:
                    for item in chunk:
                        out.write(json.dumps(item) + '\n')
                logger.info(f"Wrote chunk {chunk_num} to {output_file}")
                chunk = []
                chunk_num += 1
            total_lines += 1

        if chunk:
            output_file = f"{output_prefix}_{chunk_num}.jsonl.gz"
            with fs.open(output_file, 'wt') as out:
                for item in chunk:
                    out.write(json.dumps(item) + '\n')
            logger.info(f"Wrote final chunk {chunk_num} to {output_file}")

    logger.info(f"STEP: Finished preprocessing {input_file}. Total lines processed: {total_lines}")

def process_reviews(item):
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
    logger.info(f"STEP: Reading files with pattern: {file_pattern}")
    fs = gcsfs.GCSFileSystem()
    bag = db.read_text(file_pattern, storage_options={'fs': fs}).map(json.loads)
    logger.info(f"STEP: Sampling raw data before processing")
    sample_raw = bag.take(5)
    logger.info(f"Sample of raw data: {sample_raw}")
    logger.info(f"STEP: Processing data with {process_func.__name__}")
    processed_bag = bag.map(process_func).filter(lambda x: x is not None)
    logger.info(f"STEP: Converting processed data to Dask DataFrame")
    return processed_bag.to_dataframe()

def sample_data(df):
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
    client = setup_dask_client()
    try:
        logger.info(f"STEP: Starting data processing for category: {category_name}")
        reviews_pattern = f"{GCS_SAMPLED_DATA_PATH}/{category_name}_reviews_*.jsonl.gz"
        meta_pattern = f"{GCS_SAMPLED_DATA_PATH}/{category_name}_meta_*.jsonl.gz"

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
        output_2018_2019 = f"{GCS_SAMPLED_DATA_PATH}/sampled_data_2018_2019_{category_name}.csv"
        output_2020 = f"{GCS_SAMPLED_DATA_PATH}/sampled_data_2020_{category_name}.csv"

        fs = gcsfs.GCSFileSystem()
        with fs.open(output_2018_2019, 'w') as f:
            sampled_2018_2019_df.to_csv(f, index=False)
        with fs.open(output_2020, 'w') as f:
            sampled_2020_df.to_csv(f, index=False)

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