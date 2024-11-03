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

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,         # Send email on failure
    'email_on_retry': True,           # Send email on retry
    'email_on_success': False,        # Optional: email on success
    'email': 'vallimeenaavellaiyan@gmail.com'  # Global recipient for all tasks
}
###########need to change here ############
# Load environment variables from .env file
load_dotenv()
data_file  = os.getenv('FILE_PATH')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def update_results_xcom(ti, function_name, row_indices=None, status=None):
    """Push task results to XCom, allowing `None` values for `row_indices` or `status`."""
    ti.xcom_push(key=f"{function_name}_results", value={
        'function': function_name,
        'row_indices': row_indices,
        'status': status
    })


# Define task functions
def schema_validation_task(ti):
    logging.info("Starting schema validation task")
    df = pd.read_csv(data_file)
    status = validate_schema(df)
    
    if not status:
        logging.error("Schema validation failed.")
        raise ValueError("Schema validation failed due to column type mismatch.")
    
    update_results_xcom(ti, 'schema_validation', None, status)
    logging.info("Schema validation completed with status: %s", status)


def range_check_task(ti):
    logging.info("Starting range check task")
    df = pd.read_csv(data_file)
    rows, status = check_range(df)
    update_results_xcom(ti, 'range_check', rows, status)
    logging.info("Range check completed with status: %s", status)

def missing_duplicates_task(ti):
    logging.info("Starting missing and duplicates check task")
    df = pd.read_csv(data_file)
    missing_rows, duplicate_rows, status = find_missing_and_duplicates(df)
    update_results_xcom(ti, 'missing_duplicates', missing_rows + duplicate_rows, status)
    logging.info("Missing and duplicates check completed with status: %s", status)

def privacy_compliance_task(ti):
    logging.info("Starting privacy compliance task")
    df = pd.read_csv(data_file)
    rows, status = check_data_privacy(df)
    update_results_xcom(ti, 'privacy_compliance', rows, status)
    logging.info("Privacy compliance check completed with status: %s", status)

def emoji_detection_task(ti):
    logging.info("Starting emoji detection task")
    df = pd.read_csv(data_file)
    rows, status = detect_emoji(df)
    update_results_xcom(ti, 'emoji_detection', rows, status)
    logging.info("Emoji detection completed with status: %s", status)

def anomaly_detection_task(ti):
    logging.info("Starting anomaly detection task")
    df = pd.read_csv(data_file)
    status = detect_anomalies(df)
    update_results_xcom(ti, 'anomaly_detection', None, status)
    logging.info("Anomaly detection completed with status: %s", status)

def special_characters_detection_task(ti):
    logging.info("Starting special characters detection task")
    df = pd.read_csv(data_file)
    rows = check_only_special_characters(df)
    status = False if rows else True
    update_results_xcom(ti, 'special_characters_detection', rows, status)
    logging.info("Special characters detection completed with status: %s", status)

def review_length_checker_task(ti):
    logging.info("Starting review length check task")
    df = pd.read_csv(data_file)
    
    short_reviews, long_reviews, short_titles, long_titles, \
    short_reviews_flag, long_reviews_flag, short_titles_flag, long_titles_flag = check_review_title_length(df)

    # Store results in XCom individually, passing None for `status` if not required
    update_results_xcom(ti, 'short_reviews', short_reviews, short_reviews_flag)
    update_results_xcom(ti, 'long_reviews', long_reviews, long_reviews_flag)
    update_results_xcom(ti, 'short_titles', short_titles, short_titles_flag)
    update_results_xcom(ti, 'long_titles', long_titles, long_titles_flag)

    logging.info("Review length check completed.")

def save_results(ti):
    """Collect and save results from XCom to a CSV file."""
    # Define all individual task components to retrieve from XCom
    review_length_components = [
        'short_reviews', 'long_reviews', 'short_titles', 'long_titles'
    ]
    review_length_flags = [
        'short_reviews_flag', 'long_reviews_flag', 'short_titles_flag', 'long_titles_flag'
    ]
    
    # Other tasks to retrieve
    other_task_names = [
        'schema_validation', 'range_check', 'missing_duplicates',
        'privacy_compliance', 'emoji_detection', 'anomaly_detection',
        'special_characters_detection'
    ]

    results = []

    # Process review length components
    for component in review_length_components:
        task_result = ti.xcom_pull(key=f"{component}_results", task_ids='review_length_checker')
        if task_result:
            results.append({
                "function": component,
                "row_indices": str(task_result.get("row_indices", "")),  # Convert list to string if present
                "status": task_result.get("status", "")  # Capture status if present
            })
        else:
            logging.warning(f"No result found for component: {component}")

    # Process review length flags
    for flag in review_length_flags:
        task_result = ti.xcom_pull(key=f"{flag}_results", task_ids='review_length_checker')
        if task_result:
            results.append({
                "function": flag,
                "row_indices": "",  # No row indices for flags
                "status": task_result.get("status", "")  # Capture status if present
            })
        else:
            logging.warning(f"No result found for flag: {flag}")

    # Process other tasks
    for task in other_task_names:
        task_result = ti.xcom_pull(key=f"{task}_results", task_ids=task)
        if task_result:
            results.append({
                "function": task,
                "row_indices": str(task_result.get("row_indices", "")),  # Convert list to string if present
                "status": task_result.get("status", "")  # Capture status if present
            })
        else:
            logging.warning(f"No result found for task: {task}")

    # Convert the list of results to a DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("/opt/airflow/data/validation_results.csv", index=False)
    logging.info("Results saved successfully to validation_results.csv")

# Define the DAG
with DAG(
    dag_id='data_validation_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='DAG to perform data validation checks',
) as dag:

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=lambda: None,
    )

    schema_validation = PythonOperator(
        task_id='schema_validation',
        python_callable=schema_validation_task,
    )

    range_check = PythonOperator(
        task_id='range_check',
        python_callable=range_check_task,
    )

    missing_duplicates = PythonOperator(
        task_id='missing_duplicates',
        python_callable=missing_duplicates_task,
    )

    privacy_compliance = PythonOperator(
        task_id='privacy_compliance',
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

    final_task = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
        trigger_rule='all_success',  # Ensures save_results runs only if all previous tasks succeed
    )

        # Task to trigger the preprocessing DAG
    trigger_preprocessing_dag = TriggerDagRunOperator(
        task_id='trigger_preprocessing_dag',
        trigger_dag_id='data_preprocessing_dag',  # Replace with the actual DAG ID of your preprocessing DAG
        dag=dag,
    )

    parallel_tasks = [
        schema_validation, range_check, missing_duplicates,
        privacy_compliance, emoji_detection, anomaly_detection,
        special_characters_detection, review_length_checker
    ]
    
    load_data >> parallel_tasks >> final_task>> trigger_preprocessing_dag