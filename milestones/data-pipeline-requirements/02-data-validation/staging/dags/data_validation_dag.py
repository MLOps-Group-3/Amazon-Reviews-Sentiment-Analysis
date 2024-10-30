from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import validation functions
from utils.schema_validation import validate_schema
from utils.range_checker import check_range
from utils.missing_duplicates_checker import find_missing_and_duplicates
from utils.privacy_compliance import check_data_privacy
from utils.emoji_detection import detect_emoji
from utils.special_characters_detector import check_only_special_characters
from utils.review_length_checker import check_review_title_length
from utils.anomaly_detector import detect_anomalies

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# File path of the data to validate
data_file = "/opt/airflow/data/sampled_data_2018_2019.csv"

# DataFrame to track function names and rows with issues
results_df = pd.DataFrame(columns=['function', 'row_indices', 'validation_status'])

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_results():
    """Save the results DataFrame to a local file."""
    global results_df
    results_df.to_csv("/opt/airflow/data/validation_results.csv", index=False)
    logging.info("Results saved successfully!")

def update_results(function_name, row_indices, status):
    """Update the results DataFrame with function results."""
    global results_df
    new_entry = pd.DataFrame({
        'function': [function_name], 
        'row_indices': [row_indices], 
        'validation_status': [status]
    })
    results_df = pd.concat([results_df, new_entry], ignore_index=True)

def schema_validation_task():
    """Task to perform schema validation."""
    logging.info("Starting schema validation task")
    df = pd.read_csv(data_file)
    status = validate_schema(df)
    update_results('schema_validation', None, status)
    logging.info("Schema validation completed with status: %s", status)

def range_check_task():
    """Task to perform range check."""
    logging.info("Starting range check task")
    df = pd.read_csv(data_file)
    rows, status = check_range(df)
    update_results('range_check', rows, status)
    logging.info("Range check completed with status: %s", status)

def missing_duplicates_task():
    """Task to check for missing values and duplicates."""
    logging.info("Starting missing and duplicates check task")
    df = pd.read_csv(data_file)
    missing_rows, duplicate_rows, status = find_missing_and_duplicates(df)
    update_results('missing_values', missing_rows, status)
    update_results('duplicate_rows', duplicate_rows, status)
    logging.info("Missing and duplicates check completed with status: %s", status)

def privacy_compliance_task():
    """Task to check for privacy compliance."""
    logging.info("Starting privacy compliance task")
    df = pd.read_csv(data_file)
    rows, status = check_data_privacy(df)
    update_results('privacy_compliance', rows, status)
    logging.info("Privacy compliance check completed with status: %s", status)

def emoji_detection_task():
    """Task to detect emojis."""
    logging.info("Starting emoji detection task")
    df = pd.read_csv(data_file)
    rows, status = detect_emoji(df)
    update_results('emoji_detection', rows, status)
    logging.info("Emoji detection completed with status: %s", status)

def anomaly_detection_task():
    """Task to detect anomalies."""
    logging.info("Starting anomaly detection task")
    df = pd.read_csv(data_file)
    status = detect_anomalies(df)
    update_results('anomaly_detection', None, status)
    logging.info("Anomaly detection completed with status: %s", status)

def special_characters_detection_task():
    """Task to detect special characters."""
    logging.info("Starting special characters detection task")
    df = pd.read_csv(data_file)
    status = check_only_special_characters(df)
    update_results('special_characters_detection', None, status)
    logging.info("Special characters detection completed with status: %s", status)

def review_length_checker_task():
    """Task to check review title length."""
    logging.info("Starting review length check task")
    df = pd.read_csv(data_file)
    status = check_review_title_length(df)
    update_results('review_title_length', None, status)
    logging.info("Review length check completed with status: %s", status)

# Define the DAG
with DAG(
    dag_id='data_validation_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='DAG to perform data validation checks',
) as dag:

    # Task 1: Schema Validation (Linear)
    schema_validation = PythonOperator(
        task_id='schema_validation',
        python_callable=schema_validation_task,
    )

    # Parallel Tasks
    range_check = PythonOperator(
        task_id='range_check',
        python_callable=range_check_task,
    )

    missing_duplicates = PythonOperator(
        task_id='missing_duplicates_check',
        python_callable=missing_duplicates_task,
    )

    privacy_compliance = PythonOperator(
        task_id='privacy_compliance_check',
        python_callable=privacy_compliance_task,
    )

    emoji_detection = PythonOperator(
        task_id='emoji_detection',
        python_callable=emoji_detection_task,
    )

    anomaly_detection = PythonOperator(
        task_id='anomaly_detection',
        python_callable=anomaly_detection_task,
    )

    special_characters_detection = PythonOperator(
        task_id='special_characters_detection',
        python_callable=special_characters_detection_task,
    )

    review_length_checker = PythonOperator(
        task_id='review_length_checker',
        python_callable=review_length_checker_task,
    )

    # Task 9: Final Task (Save Results)
    final_task = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
        trigger_rule='all_done',  # Runs even if some tasks fail
    )

    # Define dependencies
    parallel_tasks = [
        range_check,
        missing_duplicates,
        privacy_compliance,
        emoji_detection,
        anomaly_detection,
        special_characters_detection,
        review_length_checker,
    ]
    
    schema_validation >> parallel_tasks >> final_task
