# Import necessary libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from airflow.operators.dagrun_operator import TriggerDagRunOperator

# Import validation functions
from utils.data_validation.schema_validation import validate_schema
from utils.data_validation.range_checker import check_range
from utils.data_validation.missing_duplicates_checker import find_missing_and_duplicates
from utils.data_validation.privacy_compliance import check_data_privacy
from utils.data_validation.emoji_detection import detect_emoji
from utils.data_validation.special_characters_detector import check_only_special_characters
from utils.data_validation.review_length_checker import check_review_title_length
from utils.data_validation.anomaly_detector import detect_anomalies

from utils.config import VALIDATION_RESULT_DATA_PATH

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': 'vallimeenaavellaiyan@gmail.com',
}

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Utility function to dynamically validate either training or serving data
def validate_data(file_path, dataset_type, ti):
    """
    Run all validation checks on a dataset and save results.
    """
    logging.info(f"Validating {dataset_type} data from file: {file_path}")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Run validation functions
    validation_results = {
        "schema_validation": validate_schema(df),
        "range_check": check_range(df),
        "missing_duplicates": find_missing_and_duplicates(df),
        "privacy_compliance": check_data_privacy(df),
        "emoji_detection": detect_emoji(df),
        "special_characters_detection": check_only_special_characters(df),
        "review_length_check": check_review_title_length(df),
        "anomaly_detection": detect_anomalies(df),
    }

    # Save validation results into XCom for tracking
    ti.xcom_push(key=f"{dataset_type}_validation_results", value=validation_results)

    # Log results
    for check, result in validation_results.items():
        logging.info(f"{check}: {'Passed' if result else 'Failed'}")

    # Raise an error if any check fails
    if not all(validation_results.values()):
        raise ValueError(f"{dataset_type} data validation failed. Check logs for details.")

def save_combined_results(ti):
    """
    Collect and save validation results from XCom for both datasets.
    """
    # Retrieve validation results from XCom
    training_results = ti.xcom_pull(key="training_validation_results")
    serving_results = ti.xcom_pull(key="serving_validation_results")

    # Prepare results for saving
    combined_results = {
        "training": training_results,
        "serving": serving_results,
    }

    # Save to a CSV file
    results_dir = os.path.dirname(VALIDATION_RESULT_DATA_PATH)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "combined_validation_results.csv")

    # Convert to DataFrame and save
    pd.DataFrame.from_dict(combined_results, orient='index').to_csv(results_file, index=True)
    logging.info(f"Validation results saved successfully to {results_file}")

# Define the DAG
with DAG(
    dag_id='03_data_validation_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Combined DAG to validate training and serving data',
) as dag:

    # Validate training data
    validate_training_task = PythonOperator(
        task_id='validate_training_data',
        python_callable=validate_data,
        op_kwargs={
            'file_path': '/opt/airflow/data/sampled/training/concatenated_training_data.csv',
            'dataset_type': 'training',
        },
    )

    # Validate serving data
    validate_serving_task = PythonOperator(
        task_id='validate_serving_data',
        python_callable=validate_data,
        op_kwargs={
            'file_path': '/opt/airflow/data/sampled/serving/concatenated_serving_data.csv',
            'dataset_type': 'serving',
        },
    )

    # Save combined validation results
    save_results_task = PythonOperator(
        task_id='save_combined_results',
        python_callable=save_combined_results,
    )

    # Trigger downstream DAG (e.g., preprocessing DAG)
    trigger_preprocessing_dag = TriggerDagRunOperator(
        task_id='trigger_preprocessing_dag',
        trigger_dag_id='04_data_preprocessing_dag',
        wait_for_completion=False,
    )

    # Set task dependencies
    [validate_training_task, validate_serving_task] >> save_results_task >> trigger_preprocessing_dag