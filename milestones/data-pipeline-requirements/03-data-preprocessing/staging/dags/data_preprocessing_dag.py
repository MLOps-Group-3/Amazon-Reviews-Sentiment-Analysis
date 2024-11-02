from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import utility functions
from utils.data_cleaning_pandas import clean_amazon_reviews
from utils.data_labeling import apply_labelling
from utils.aspect_extraction import tag_and_expand_aspects, get_synonyms
# from utils.aspect_extraction_parallel import parallel_process_aspects
from utils.aspect_data_labeling import apply_vader_labeling

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 30),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# File paths
data_file = "/opt/airflow/data/sampled_data_2018_2019.csv"
cleaned_data_file = "/opt/airflow/data/airflow/cleaned_data.csv"
validation_data_file = "/opt/airflow/data/validation_data/validation_results.csv"
labeled_data_file = "/opt/airflow/data/airflow/labeled_data.csv"
aspect_data_file = "/opt/airflow/data/cleaned_data/aspect_extracted_data.csv"
labeled_aspect_data_file = "/opt/airflow/data/airflow/labeled_aspect_data.csv"

# Define tasks
def data_cleaning_task():
    """Task to perform data cleaning."""
    try:
        logger.info("Starting data cleaning task...")
        
        # Load raw data and validation data
        df = pd.read_csv(data_file)
        logger.info(f"Loaded raw data with shape: {df.shape} from {data_file}")
        
        validation_df = pd.read_csv(validation_data_file)
        logger.info(f"Loaded validation data with shape: {validation_df.shape} from {validation_data_file}")
        
        # Extract emoji indices
        emoji_indices = eval(validation_df.loc[validation_df["function"] == "emoji_detection", "row_indices"].values[0])
        logger.info(f"Extracted emoji indices: {emoji_indices[:10]}... (truncated for brevity)")

        # Clean data
        df_cleaned = clean_amazon_reviews(df, emoji_indices)
        logger.info(f"Data cleaning completed. Cleaned data shape: {df_cleaned.shape}")

        # Save cleaned data
        df_cleaned.to_csv(cleaned_data_file, index=False)
        logger.info(f"Cleaned data saved to {cleaned_data_file}")
        
    except Exception as e:
        logger.error("Error during data cleaning task.", exc_info=True)
        raise e

def data_labeling_task():
    """Task to perform data labeling."""
    try:
        logger.info("Starting data labeling task...")
        
        # Load cleaned data
        df = pd.read_csv(cleaned_data_file)
        logger.info(f"Loaded cleaned data with shape: {df.shape} from {cleaned_data_file}")

        # Apply labeling
        df_labeled = apply_labelling(df)
        logger.info(f"Data labeling completed. Labeled data shape: {df_labeled.shape}")

        # Save labeled data
        df_labeled.to_csv(labeled_data_file, index=False)
        logger.info(f"Labeled data saved to {labeled_data_file}")
        
    except Exception as e:
        logger.error("Error during data labeling task.", exc_info=True)
        raise e
    
def aspect_extraction_task():
    """Task to perform aspect extraction."""
    try:
        logger.info("Starting aspect extraction task...")
        
        # Load cleaned data
        df = pd.read_csv(cleaned_data_file)
        logger.info(f"Loaded cleaned data with shape: {df.shape} from {cleaned_data_file}")

        aspects = {
            "delivery": get_synonyms("delivery") | {"arrive", "shipping"},
            "quality": get_synonyms("quality") | {"craftsmanship", "durable"},
            "customer_service": get_synonyms("service") | {"support", "helpful", "response"},
            "product_design": get_synonyms("design") | {"appearance", "look", "style"},
            "cost": get_synonyms("cost") | get_synonyms("price") | {"value", "expensive", "cheap", "affordable"}
            }

        # Apply extraction
        df_aspect = tag_and_expand_aspects(df,aspects)
        # df_aspect = parallel_process_aspects(df,aspects)
        logger.info(f"Data labeling completed. Labeled data shape: {df_aspect.shape}")

        # Save labeled data
        df_aspect.to_csv(aspect_data_file, index=False)
        logger.info(f"Labeled data saved to {aspect_data_file}")
        
    except Exception as e:
        logger.error("Error during data labeling task.", exc_info=True)
        raise e



def data_labeling_aspect_task():
    """Task to perform data labeling."""
    try:
        logger.info("Starting data labeling task...")
        
        # Load cleaned data
        df = pd.read_csv(aspect_data_file)
        logger.info(f"Loaded cleaned aspect data with shape: {df.shape} from {aspect_data_file}")

        # Apply labeling
        df_labeled = apply_vader_labeling(df)
        logger.info(f"Aspect Data labeling completed. Labeled data shape: {df_labeled.shape}")

        # Save labeled data
        df_labeled.to_csv(labeled_aspect_data_file, index=False)
        logger.info(f"Labeled data saved to {labeled_aspect_data_file}")
        
    except Exception as e:
        logger.error("Error during aspect data labeling task.", exc_info=True)
        raise e



# Define the DAG
with DAG(
    dag_id='data_preprocessing_dag',
    default_args=default_args,
    schedule_interval='@daily',
    description='DAG for data cleaning and labeling',
) as dag:

    # Task 1: Data Cleaning
    data_cleaning = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning_task,
    )

    # Task 2.1: Data Labeling for overall sentiment
    data_labeling = PythonOperator(
        task_id='data_labeling',
        python_callable=data_labeling_task,
    )

    # Task 2.2: Aspect Capturing

    aspect_extraction = PythonOperator(
        task_id='aspect_extraction',
        python_callable=aspect_extraction_task,
    )

    # Task 3: Aspect Data Labeling for aspect based sentiment
    data_labeling_aspect = PythonOperator(
        task_id='data_labeling_aspect',
        python_callable=data_labeling_aspect_task,
    )

    data_cleaning >> [aspect_extraction, data_labeling] >> data_labeling_aspect
