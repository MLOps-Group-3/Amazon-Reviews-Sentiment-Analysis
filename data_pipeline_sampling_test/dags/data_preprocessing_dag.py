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
    CLEANED_DATA_PATH_TRAIN,
    CLEANED_DATA_PATH_SERVE,
    CLEANED_ASPECT_DATA_PATH_TRAIN,
    CLEANED_ASPECT_DATA_PATH_SERVE,
    LABELED_DATA_PATH_TRAIN,
    LABELED_ASPECT_DATA_PATH_TRAIN,
    SAMPLED_TRAINING_DIRECTORY,
    SAMPLED_SERVING_DIRECTORY,
    VALIDATION_RESULT_TRAINING_DATA_PATH,
    VALIDATION_RESULT_SERVING_DATA_PATH,
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

def get_latest_file_with_prefix(directory, prefix):
    """Get the latest file in a directory with a specific prefix."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No CSV files found with prefix '{prefix}' in directory: {directory}")
    return max(files, key=os.path.getmtime)

def get_validation_file(directory):
    """Get the validation_results.csv file from the specified directory."""
    file_path = os.path.join(directory, "validation_results.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"validation_results.csv not found in directory: {directory}")
    return file_path

# Define task functions
def data_cleaning_task(**kwargs):
    """Task to perform data cleaning."""
    try:
        triggering_dag_id = kwargs['dag_run'].conf.get('triggering_dag_id')
        logger.info(f"Starting data cleaning task. Triggered by: {triggering_dag_id}")
        
        # Select the correct directories and paths based on the triggering DAG
        if triggering_dag_id == '03_sampling_train_dag':
            data_directory = SAMPLED_TRAINING_DIRECTORY
            validation_directory = VALIDATION_RESULT_TRAINING_DATA_PATH
            cleaned_data_path = CLEANED_DATA_PATH_TRAIN
            cleaned_aspect_data_path = CLEANED_ASPECT_DATA_PATH_TRAIN
        elif triggering_dag_id == '03_sampling_serve_dag':
            data_directory = SAMPLED_SERVING_DIRECTORY
            validation_directory = VALIDATION_RESULT_SERVING_DATA_PATH
            cleaned_data_path = CLEANED_DATA_PATH_SERVE
            cleaned_aspect_data_path = CLEANED_ASPECT_DATA_PATH_SERVE
        else:
            raise ValueError(f"Unknown triggering DAG ID: {triggering_dag_id}")
        
        # Get the latest concatenated data file
        data_file = get_latest_file_with_prefix(data_directory, 'concatenated_')
        df = pd.read_csv(data_file)
        logger.info(f"Loaded raw data with shape: {df.shape} from {data_file}")
        log_data_card(df, "Raw Data")
        
        # Get the validation file
        validation_file = get_validation_file(validation_directory)
        validation_df = pd.read_csv(validation_file)
        logger.info(f"Loaded validation data with shape: {validation_df.shape} from {validation_file}")
        
        # Extract emoji indices for cleaning
        emoji_indices = eval(validation_df.loc[validation_df["function"] == "emoji_detection", "row_indices"].values[0])
        df_cleaned = clean_amazon_reviews(df, emoji_indices)
        
        # Save cleaned data
        os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
        df_cleaned.to_csv(cleaned_data_path, index=False)
        logger.info(f"Cleaned data saved to {cleaned_data_path}")
        
        # Store the cleaned data path and triggering DAG ID for downstream tasks
        kwargs['ti'].xcom_push(key='cleaned_data_path', value=cleaned_data_path)
        kwargs['ti'].xcom_push(key='cleaned_aspect_data_path', value=cleaned_aspect_data_path)
        kwargs['ti'].xcom_push(key='triggering_dag_id', value=triggering_dag_id)
        
    except Exception as e:
        logger.error("Error during data cleaning task.", exc_info=True)
        raise e

def aspect_extraction_task(**kwargs):
    """Task to perform aspect extraction."""
    try:
        logger.info("Starting aspect extraction task...")
        cleaned_data_path = kwargs['ti'].xcom_pull(task_ids='data_cleaning', key='cleaned_data_path')
        cleaned_aspect_data_path = kwargs['ti'].xcom_pull(task_ids='data_cleaning', key='cleaned_aspect_data_path')
        
        df = pd.read_csv(cleaned_data_path)
        aspects = {
            "delivery": get_synonyms("delivery") | {"arrive", "shipping"},
            "quality": get_synonyms("quality") | {"craftsmanship", "durable"},
            "customer_service": get_synonyms("service") | {"support", "helpful", "response"},
            "product_design": get_synonyms("design") | {"appearance", "look", "style"},
            "cost": get_synonyms("cost") | get_synonyms("price") | {"value", "expensive", "cheap", "affordable"},
        }
        df_aspect = tag_and_expand_aspects(df, aspects)
        
        os.makedirs(os.path.dirname(cleaned_aspect_data_path), exist_ok=True)
        df_aspect.to_csv(cleaned_aspect_data_path, index=False)
        log_data_card(df_aspect, "Aspect Extracted Data")
        
    except Exception as e:
        logger.error("Error during aspect extraction task.", exc_info=True)
        raise e

def data_labeling_task(**kwargs):
    """Task to perform data labeling."""
    try:
        logger.info("Starting data labeling task...")
        cleaned_data_path = kwargs['ti'].xcom_pull(task_ids='data_cleaning', key='cleaned_data_path')
        df = pd.read_csv(cleaned_data_path)
        df_labeled = apply_labelling(df)
        
        os.makedirs(os.path.dirname(LABELED_DATA_PATH_TRAIN), exist_ok=True)
        df_labeled.to_csv(LABELED_DATA_PATH_TRAIN, index=False)
        log_data_card(df_labeled, "Labeled Data")
    except Exception as e:
        logger.error("Error during data labeling task.", exc_info=True)
        raise e

def data_labeling_aspect_task(**kwargs):
    """Task to perform aspect-based data labeling."""
    try:
        logger.info("Starting aspect-based data labeling task...")
        cleaned_aspect_data_path = kwargs['ti'].xcom_pull(task_ids='data_cleaning', key='cleaned_aspect_data_path')
        df = pd.read_csv(cleaned_aspect_data_path)
        df_labeled = apply_vader_labeling(df)
        
        os.makedirs(os.path.dirname(LABELED_ASPECT_DATA_PATH_TRAIN), exist_ok=True)
        df_labeled.to_csv(LABELED_ASPECT_DATA_PATH_TRAIN, index=False)
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
        """Check if the triggering DAG is for training."""
        return kwargs['ti'].xcom_pull(task_ids='data_cleaning', key='triggering_dag_id') == '03_sampling_train_dag'

    # Task 1: Data Cleaning
    data_cleaning = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning_task,
        provide_context=True,
    )

    # Task 2: Aspect Extraction
    aspect_extraction = PythonOperator(
        task_id='aspect_extraction',
        python_callable=aspect_extraction_task,
        provide_context=True,
    )

    # Branching task to decide whether to run labeling
    branching = BranchPythonOperator(
        task_id='branching_task',
        python_callable=lambda **kwargs: "run_labeling" if is_training(**kwargs) else "skip_labeling",
        provide_context=True,
    )

    # Task 3: Data Labeling for overall sentiment (only for training)
    data_labeling = PythonOperator(
        task_id='data_labeling',
        python_callable=data_labeling_task,
        provide_context=True,
    )

    # Task 4: Aspect Data Labeling for aspect-based sentiment (only for training)
    data_labeling_aspect = PythonOperator(
        task_id='data_labeling_aspect',
        python_callable=data_labeling_aspect_task,
        provide_context=True,
    )

    # Dummy tasks for branching
    run_labeling = BashOperator(
        task_id='run_labeling',
        bash_command="echo 'Running labeling tasks for training data.'",
    )

    skip_labeling = BashOperator(
        task_id='skip_labeling',
        bash_command="echo 'Skipping labeling tasks for serving data.'",
    )

    # Task dependencies
    data_cleaning >> aspect_extraction >> branching
    branching >> run_labeling >> [data_labeling, data_labeling_aspect]
    branching >> skip_labeling
