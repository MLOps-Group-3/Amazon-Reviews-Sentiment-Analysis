from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
# Import utility functions
from utils.data_preprocessing.data_cleaning_pandas import clean_amazon_reviews
from utils.data_preprocessing.data_labeling import apply_labelling
from utils.data_preprocessing.aspect_extraction import tag_and_expand_aspects, get_synonyms
from utils.data_preprocessing.aspect_data_labeling import apply_vader_labeling
from utils.config import (
    TRAINING_SAMPLED_DATA_PATH, SERVING_SAMPLED_DATA_PATH,
    CLEANED_DATA_PATH, CLEANED_ASPECT_DATA_PATH,
    LABELED_DATA_PATH, LABELED_ASPECT_DATA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': 'vallimeenaavellaiyan@gmail.com',
}

def process_data(file_path, cleaned_file_path, aspect_file_path, labeled_file_path, labeled_aspect_file_path):
    try:
        # Load sampled data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")

        # Step 1: Data Cleaning
        df_cleaned = clean_amazon_reviews(df)
        logger.info(f"Cleaned data shape: {df_cleaned.shape}")
        os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
        df_cleaned.to_csv(cleaned_file_path, index=False)

        # Step 2: Aspect Extraction
        aspects = {
            "delivery": get_synonyms("delivery"),
            "quality": get_synonyms("quality"),
            # Add more aspects if needed
        }
        df_aspect = tag_and_expand_aspects(df_cleaned, aspects)
        logger.info(f"Aspect extraction completed with shape: {df_aspect.shape}")
        os.makedirs(os.path.dirname(aspect_file_path), exist_ok=True)
        df_aspect.to_csv(aspect_file_path, index=False)

        # Step 3: Data Labeling
        df_labeled = apply_labelling(df_cleaned)
        logger.info(f"Labeled data shape: {df_labeled.shape}")
        os.makedirs(os.path.dirname(labeled_file_path), exist_ok=True)
        df_labeled.to_csv(labeled_file_path, index=False)

        # Step 4: Aspect-Based Data Labeling
        df_aspect_labeled = apply_vader_labeling(df_aspect)
        logger.info(f"Aspect-labeled data shape: {df_aspect_labeled.shape}")
        os.makedirs(os.path.dirname(labeled_aspect_file_path), exist_ok=True)
        df_aspect_labeled.to_csv(labeled_aspect_file_path, index=False)
    except Exception as e:
        logger.error("Error in processing data.", exc_info=True)
        raise e

with DAG(
    dag_id='04_data_preprocessing_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    preprocess_training_data = PythonOperator(
        task_id='preprocess_training_data',
        python_callable=process_data,
        op_kwargs={
            'file_path': TRAINING_SAMPLED_DATA_PATH,
            'cleaned_file_path': CLEANED_DATA_PATH,
            'aspect_file_path': CLEANED_ASPECT_DATA_PATH,
            'labeled_file_path': LABELED_DATA_PATH,
            'labeled_aspect_file_path': LABELED_ASPECT_DATA_PATH,
        },
    )

    preprocess_serving_data = PythonOperator(
        task_id='preprocess_serving_data',
        python_callable=process_data,
        op_kwargs={
            'file_path': SERVING_SAMPLED_DATA_PATH,
            'cleaned_file_path': CLEANED_DATA_PATH,
            'aspect_file_path': CLEANED_ASPECT_DATA_PATH,
            'labeled_file_path': LABELED_DATA_PATH,
            'labeled_aspect_file_path': LABELED_ASPECT_DATA_PATH,
        },
    )

    preprocess_training_data >> preprocess_serving_data
