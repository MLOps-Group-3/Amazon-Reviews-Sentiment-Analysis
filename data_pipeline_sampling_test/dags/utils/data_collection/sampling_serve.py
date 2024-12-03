import os
import pandas as pd
import gzip
import json
import calendar
import logging
from tqdm import tqdm
from datetime import datetime
from utils.data_collection.dynamic_month import get_next_serving_month
from utils.config import CATEGORIES, TARGET_DIRECTORY, SAMPLED_SERVING_DIRECTORY, SAMPLING_FRACTION

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def read_jsonl_gz_in_chunks(file_path, chunksize=10000):
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
    reviews_df['review_date_timestamp'] = pd.to_datetime(reviews_df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    start_date = f"{year}-{month:02d}-01 00:00:00"
    _, last_day = calendar.monthrange(year, month)
    end_date = f"{year}-{month:02d}-{last_day:02d} 23:59:59"
    logger.info(f"filtering dara from {start_date} to {end_date}")
    filtered_reviews_df = reviews_df[
        (reviews_df['review_date_timestamp'] >= start_date) & 
        (reviews_df['review_date_timestamp'] <= end_date)
    ].drop(columns=['images'], errors='ignore')
    return filtered_reviews_df

def join_dataframes(filtered_reviews_df, metadata_df):
    metadata_df = metadata_df.rename(columns={'title': 'product_name'})
    metadata_df = metadata_df[['parent_asin', 'main_category', 'product_name', 'categories', 'price', 'average_rating', 'rating_number']]
    return filtered_reviews_df.merge(metadata_df, on='parent_asin', how='left')

def sample_data(joined_df):
    joined_df['review_month'] = pd.to_datetime(joined_df['review_date_timestamp']).dt.month
    sampled_df = joined_df.groupby(['review_month', 'rating']).apply(
        lambda x: x.sample(frac=SAMPLING_FRACTION, random_state=42)
    ).reset_index(drop=True)
    return sampled_df

def save_sampled_data(sampled_df, category_name, year, month):
    os.makedirs(SAMPLED_SERVING_DIRECTORY, exist_ok=True)
    file_path = os.path.join(SAMPLED_SERVING_DIRECTORY, f"sampled_data_{year}_{month:02d}_{category_name}.csv")
    sampled_df.to_csv(file_path, index=False)
    logger.info(f"Saved sampled data to {file_path}")

def sample_serving_data(category_name, year, month):
    try:
        reviews_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_reviews.jsonl.gz")
        meta_file = os.path.join(TARGET_DIRECTORY, f"{category_name}_meta.jsonl.gz")

        reviews_df = load_jsonl_gz(reviews_file)
        metadata_df = load_jsonl_gz(meta_file)
        filtered_reviews_df = process_reviews_df(reviews_df, year, month)
        joined_df = join_dataframes(filtered_reviews_df, metadata_df)
        sampled_df = sample_data(joined_df)
        save_sampled_data(sampled_df, category_name, year, month)
    except Exception as e:
        logger.error(f"Error while processing category {category_name}: {e}")
