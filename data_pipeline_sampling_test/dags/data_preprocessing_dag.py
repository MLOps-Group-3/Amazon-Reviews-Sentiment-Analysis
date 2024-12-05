from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
from utils.data_preprocessing.data_cleaning_pandas import clean_amazon_reviews
from utils.data_preprocessing.data_labeling import apply_labelling
from utils.data_preprocessing.aspect_extraction import tag_and_expand_aspects, get_synonyms
from utils.data_preprocessing.aspect_data_labeling import apply_vader_labeling
from utils.config import (
    TRAINING_SAMPLED_DATA_PATH,
    SERVING_SAMPLED_DATA_PATH,
    VALIDATION_RESULT_DATA_PATH,
    CLEANED_DATA_PATH,
    CLEANED_ASPECT_DATA_PATH,
    LABELED_DATA_PATH,
    LABELED_ASPECT_DATA_PATH,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': False,
    'email': 'vallimeenaavellaiyan@gmail.com'
}

# Define utility functions
def log_data_card(df, task_name):
    """Logs basic data statistics for a DataFrame."""
    logger.info(f"{task_name} Data Card:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Sample Data:\n{df.head()}")
    logger.info(f"Data Types:\n{df.dtypes}")
    logger.info(f"Missing Values:\n{df.isnull().sum()}")

# Utility function to get the latest file
def get_latest_file(directory):
    """Get the latest file in a directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")
    return max(files, key=os.path.getmtime)

# Define task functions
def data_cleaning_task(mode):
    """Task to perform data cleaning."""
    try:
        logger.info(f"Starting data cleaning task for mode: {mode}")
        
        # Select the latest data file based on the mode
        data_directory = TRAINING_SAMPLED_DATA_PATH if mode == "training" else SERVING_SAMPLED_DATA_PATH
        data_file = get_latest_file(data_directory)
        df = pd.read_csv(data_file)
        logger.info(f"Loaded raw data with shape: {df.shape} from {data_file}")
        log_data_card(df, "Raw Data")
        
        # Select the correct validation folder based on the mode
        validation_directory = os.path.join(VALIDATION_RESULT_DATA_PATH, mode)
        validation_file = get_latest_file(validation_directory)
        validation_df = pd.read_csv(validation_file)
        logger.info(f"Loaded validation data with shape: {validation_df.shape} from {validation_file}")
        
        # Extract emoji indices for cleaning
        emoji_indices = eval(validation_df.loc[validation_df["function"] == "emoji_detection", "row_indices"].values[0])
        df_cleaned = clean_amazon_reviews(df, emoji_indices)
        
        # Save cleaned data
        os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Cleaned data saved to {CLEANED_DATA_PATH}")
    except Exception as e:
        logger.error("Error during data cleaning task.", exc_info=True)
        raise e

def data_labeling_task():
    """Task to perform data labeling."""
    try:
        logger.info("Starting data labeling task...")
        df = pd.read_csv(CLEANED_DATA_PATH)
        df_labeled = apply_labelling(df)
        os.makedirs(os.path.dirname(LABELED_DATA_PATH), exist_ok=True)
        df_labeled.to_csv(LABELED_DATA_PATH, index=False)
        log_data_card(df_labeled, "Labeled Data")
    except Exception as e:
        logger.error("Error during data labeling task.", exc_info=True)
        raise e

def aspect_extraction_task():
    """Task to perform aspect extraction."""
    try:
        logger.info("Starting aspect extraction task...")
        df = pd.read_csv(CLEANED_DATA_PATH)
        aspects = {
            "delivery": get_synonyms("delivery") | {"arrive", "shipping"},
            "quality": get_synonyms("quality") | {"craftsmanship", "durable"},
            "customer_service": get_synonyms("service") | {"support", "helpful", "response"},
            "product_design": get_synonyms("design") | {"appearance", "look", "style"},
            "cost": get_synonyms("cost") | get_synonyms("price") | {"value", "expensive", "cheap", "affordable"},
        }
        df_aspect = tag_and_expand_aspects(df, aspects)
        os.makedirs(os.path.dirname(CLEANED_ASPECT_DATA_PATH), exist_ok=True)
        df_aspect.to_csv(CLEANED_ASPECT_DATA_PATH, index=False)
        log_data_card(df_aspect, "Aspect Extracted Data")
    except Exception as e:
        logger.error("Error during aspect extraction task.", exc_info=True)
        raise e

def data_labeling_aspect_task():
    """Task to perform aspect-based data labeling."""
    try:
        logger.info("Starting aspect-based data labeling task...")
        df = pd.read_csv(CLEANED_ASPECT_DATA_PATH)
        df_labeled = apply_vader_labeling(df)
        os.makedirs(os.path.dirname(LABELED_ASPECT_DATA_PATH), exist_ok=True)
        df_labeled.to_csv(LABELED_ASPECT_DATA_PATH, index=False)
        log_data_card(df_labeled, "Aspect Labeled Data")
    except Exception as e:
        logger.error("Error during aspect data labeling task.", exc_info=True)
        raise e

# Define the DAG
with DAG(
    dag_id='04_data_preprocessing_dag',
    default_args=default_args,
    schedule_interval=None,
    description='DAG for data cleaning and labeling',
) as dag:

    def is_training(**kwargs):
        """Check if the mode is training."""
        return kwargs['dag_run'].conf.get('mode', 'training') == 'training'

    # Task 1: Data Cleaning
    data_cleaning = PythonOperator(
        task_id='data_cleaning',
        python_callable=lambda **kwargs: data_cleaning_task(kwargs['dag_run'].conf.get('mode', 'training')),
        provide_context=True,
    )

    # Task 2.1: Data Labeling for overall sentiment
    data_labeling = PythonOperator(
        task_id='data_labeling',
        python_callable=data_labeling_task,
    )

    # Task 2.2: Aspect Extraction
    aspect_extraction = PythonOperator(
        task_id='aspect_extraction',
        python_callable=aspect_extraction_task,
    )

    # Task 3: Aspect Data Labeling for aspect-based sentiment
    data_labeling_aspect = PythonOperator(
        task_id='data_labeling_aspect',
        python_callable=data_labeling_aspect_task,
    )

    # Branching task to decide whether to run labeling
    branching = BranchPythonOperator(
        task_id='branching_task',
        python_callable=lambda **kwargs: "run_labeling" if is_training(**kwargs) else "skip_labeling",
        provide_context=True,
    )

    # Dummy tasks for branching
    run_labeling = BashOperator(
        task_id='run_labeling',
        bash_command="echo 'Running labeling tasks.'",
    )

    skip_labeling = BashOperator(
        task_id='skip_labeling',
        bash_command="echo 'Skipping labeling tasks for serving data.'",
    )

    # Task dependencies
    data_cleaning >> aspect_extraction >> branching
    branching >> run_labeling >> [data_labeling, data_labeling_aspect]
    branching >> skip_labeling
