#!/usr/bin/env python
# coding: utf-8
from operator import index

# In[6]:

import pandas as pd
import json
import os
import logging
from datetime import datetime
from ..dags.utils.config import CATEGORIES, TARGET_DIRECTORY_SAMPLED, TARGET_DIRECTORY


def setup_logging():

    log_directory = "/opt/airflow/logs"
    os.makedirs(log_directory, exist_ok=True)
    initial_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"sampling_{initial_timestamp}.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(log_directory, log_file_name))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Logging configuration set up successfully.')
    return logger


def load_and_display_json(logger, category, file_type, num_rows):
    file_name = f"{category}_{file_type}.jsonl.gz"
    jsonl_gz_file = os.path.join(TARGET_DIRECTORY, file_name)
    logger.info(f"Loading JSON file: {jsonl_gz_file}")
    try:
        df = pd.read_json(jsonl_gz_file, lines=True)
        logger.info(f"Successfully read JSON file: {jsonl_gz_file}")
    except Exception as e:
        logger.error(f"Failed to read JSON file: {jsonl_gz_file}, Error: {e}")
        return None

    logger.info(f"Schema for {file_name}:")
    print(f"\nSchema for {file_name}:")
    print(df.dtypes)

    logger.info(f"Top {num_rows} rows for {file_name}:")
    print(f"\nTop {num_rows} rows for {file_name}:")
    print(df.head(num_rows))

    return df


def preprocess_reviews_data(logger, reviews_df, num_rows):
    logger.info("Preprocessing reviews data.")
    reviews_df['review_date_timestamp'] = pd.to_datetime(reviews_df['timestamp'], unit='ms').dt.strftime(
        '%Y-%m-%d %H:%M:%S')
    filtered_reviews_df = reviews_df[
        (reviews_df['review_date_timestamp'] >= '2018-01-01 00:00:00') &
        (reviews_df['review_date_timestamp'] <= '2020-12-31 23:59:59')
        ].drop(columns=['images'])

    logger.info(f"Filtered reviews data, showing top {num_rows} rows:")
    print(filtered_reviews_df.head(num_rows))
    return filtered_reviews_df


def rename_metadata_and_show(logger, metadata_df, num_rows):
    logger.info("Renaming metadata columns.")
    metadata_df_renamed = metadata_df.rename(columns={'title': 'product_name'})
    metadata_df_renamed = metadata_df_renamed[
        ['parent_asin', 'main_category', 'product_name', 'categories', 'price', 'average_rating', 'rating_number']]

    logger.info("Renamed Metadata Schema:")
    print("Renamed Metadata Schema:")
    print(metadata_df_renamed.dtypes)

    logger.info(f"Top {num_rows} rows of the renamed metadata DataFrame:")
    print(f"Top {num_rows} rows of the renamed metadata DataFrame:")
    print(metadata_df_renamed.head(num_rows))

    return metadata_df_renamed


def join_filtered_reviews_and_renamed_metadata(logger, filtered_reviews_df, renamed_metadata_df, num_rows):
    logger.info("Joining filtered reviews with renamed metadata.")
    joined_df = pd.merge(filtered_reviews_df, renamed_metadata_df, on='parent_asin', how='left')

    logger.info(f"Top {num_rows} rows of the joined DataFrame:")
    print(f"Top {num_rows} rows of the joined DataFrame:")
    print(joined_df.head(num_rows))

    filtered_row_count = len(filtered_reviews_df)
    print(f"Total number of rows in filtered reviews DataFrame: {filtered_row_count}")
    logger.info(f"Total number of rows in filtered reviews DataFrame: {filtered_row_count}")

    joined_row_count = len(joined_df)
    print(f"Row count of joined_df: {joined_row_count}")
    logger.info(f"Row count of joined_df: {joined_row_count}")

    category_counts = joined_df['main_category'].value_counts().reset_index()
    category_counts.columns = ['main_category', 'count']
    category_counts = category_counts.sort_values('count', ascending=False)

    logger.info("Distinct main categories and their row counts:")
    print("Distinct main categories and their row counts:")
    print(category_counts)

    return joined_df


def add_review_month_and_count_and_sample_joined_dataframe(logger, joined_df, sampling_fraction=0.01):
    logger.info("Adding review month and counting occurrences.")
    joined_df['review_month'] = pd.to_datetime(joined_df['review_date_timestamp']).dt.month
    grouped_df = joined_df.groupby(['review_month', 'rating']).size().reset_index(name='count')
    joined_with_count_df = pd.merge(joined_df, grouped_df, on=['review_month', 'rating'])

    sampled_df = joined_with_count_df.groupby(['review_month', 'rating']).apply(
        lambda x: x.sample(frac=sampling_fraction, random_state=42)).reset_index(drop=True)

    logger.info("Sampled DataFrame:")
    print(sampled_df.head())

    row_count = len(sampled_df)
    print(f"Row count of sampled_df: {row_count}")
    logger.info(f"Row count of sampled_df: {row_count}")

    logger.info("Data types of columns in sampled_df:")
    logger.info(sampled_df.dtypes)
    print("Data types of columns in sampled_df:")
    print(sampled_df.dtypes)

    return sampled_df


def preprocess_sampled_data(logger, sampled_df):
    logger.info("Preprocessing sampled data.")
    sampled_df['review_date_timestamp'] = pd.to_datetime(sampled_df['review_date_timestamp'])
    sampled_df['year'] = sampled_df['review_date_timestamp'].dt.year
    sampled_df['categories'] = sampled_df['categories'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    sampled_df = sampled_df.drop(columns=['count'])

    logger.info("Preprocessed sampled DataFrame:")
    print(sampled_df.head())

    return sampled_df


def filter_and_save_preprocessed_sampled_data(logger, preprocessed_sampled_df, category):
    logger.info(f"Filtering and saving preprocessed sampled data for category: {category}")
    sampled_2018_2019_df = preprocessed_sampled_df[preprocessed_sampled_df['year'].isin([2018, 2019])]
    sampled_2020_df = preprocessed_sampled_df[preprocessed_sampled_df['year'] == 2020]

    row_count_2018_2019 = len(sampled_2018_2019_df)
    print(f"Row count of sampled_2018_2019_df for {category}: {row_count_2018_2019}")
    logger.info(f"Row count of sampled_2018_2019_df for {category}: {row_count_2018_2019}")

    row_count_2020 = len(sampled_2020_df)
    print(f"Row count of sampled_2020_df for {category}: {row_count_2020}")
    logger.info(f"Row count of sampled_2020_df for {category}: {row_count_2020}")

    os.makedirs(TARGET_DIRECTORY_SAMPLED, exist_ok=True)

    sampled_2018_2019_df.to_csv(os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2018_2019_{category}.csv"), index=False)
    sampled_2020_df.to_csv(os.path.join(TARGET_DIRECTORY_SAMPLED, f"sampled_data_2020_{category}.csv"), index=False)

    logger.info(f"Data successfully saved for category {category}.")


def process_category(category):
    logger = setup_logging()
    logger.info(f"Processing category: {category}")

    # Load reviews data
    reviews_df = load_and_display_json(logger, category, "reviews", num_rows=5)

    # Load metadata
    metadata_df = load_and_display_json(logger, category, "meta", num_rows=5)

    if reviews_df is None or metadata_df is None:
        logger.error(f"Failed to load data for category: {category}")
        return

    filtered_reviews_df = preprocess_reviews_data(logger, reviews_df, num_rows=5)
    renamed_metadata_df = rename_metadata_and_show(logger, metadata_df, num_rows=5)
    joined_df = join_filtered_reviews_and_renamed_metadata(logger, filtered_reviews_df, renamed_metadata_df, num_rows=5)
    sampled_df = add_review_month_and_count_and_sample_joined_dataframe(logger, joined_df, sampling_fraction=0.01)
    preprocessed_sampled_df = preprocess_sampled_data(logger, sampled_df)
    filter_and_save_preprocessed_sampled_data(logger, preprocessed_sampled_df, category)

    logger.info(f"Processing completed for category: {category}")