# Import necessary libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator  # Updated import for Airflow 2.x
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from glob import glob

# Import validation functions
from utils.data_validation.schema_validation import validate_schema
from utils.data_validation.range_checker import check_range
from utils.data_validation.missing_duplicates_checker import find_missing_and_duplicates
from utils.data_validation.privacy_compliance import check_data_privacy
from utils.data_validation.emoji_detection import detect_emoji
from utils.data_validation.special_characters_detector import check_only_special_characters
from utils.data_validation.review_length_checker import check_review_title_length
from utils.data_validation.anomaly_detector import detect_anomalies

from utils.config import SAMPLED_TRAINING_DIRECTORY, SAMPLED_SERVING_DIRECTORY

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

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_latest_file(directory, prefix):
    """Find the latest file in a directory matching a prefix."""
    files = glob(os.path.join(directory, f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found in {directory} with prefix {prefix}")
    latest_file = max(files, key=os.path.getctime)  # Sort by creation time
    return latest_file

def get_data_path(**kwargs):
    """Determine the input data path based on the triggering DAG."""
    dag_run = kwargs.get('dag_run')
    if not dag_run:
        raise ValueError("dag_run not found in kwargs")
    conf = dag_run.conf or {}
    triggering_dag_id = conf.get('triggering_dag_id')
    if not triggering_dag_id:
        raise ValueError("triggering_dag_id not provided. Please provide 'sampling_train_dag' or 'sampling_serve_dag'.")
    logging.info(f"Triggering DAG ID: {triggering_dag_id}")
    if triggering_dag_id == 'sampling_train_dag':
        return get_latest_file(SAMPLED_TRAINING_DIRECTORY, "concatenated_training_data")
    elif triggering_dag_id == 'sampling_serve_dag':
        return get_latest_file(SAMPLED_SERVING_DIRECTORY, "concatenated_serving_data")
    else:
        raise ValueError(f"Unknown triggering DAG ID: {triggering_dag_id}")

def set_data_file(ti, **kwargs):
    data_path = get_data_path(**kwargs)
    logging.info(f"Selected data path: {data_path}")
    ti.xcom_push(key='data_file_path', value=data_path)

def update_results_xcom(ti, function_name, row_indices=None, status=None):
    """Push task results to XCom, allowing `None` values for `row_indices` or `status`."""
    ti.xcom_push(key=f"{function_name}_results", value={
        'function': function_name,
        'row_indices': row_indices,
        'status': status
    })

# Define task functions (updated to accept data_file parameter)
def schema_validation_task(ti, data_file):
    logging.info("Starting schema validation task")
    df = pd.read_csv(data_file)
    status = validate_schema(df)
    if not status:
        logging.error("Schema validation failed.")
        raise ValueError("Schema validation failed due to column type mismatch.")
    update_results_xcom(ti, 'schema_validation', None, status)
    logging.info("Schema validation completed with status: %s", status)

def range_check_task(ti, data_file):
    logging.info("Starting range check task")
    df = pd.read_csv(data_file)
    rows, status = check_range(df)
    update_results_xcom(ti, 'range_check', rows, status)
    logging.info("Range check completed with status: %s", status)

def missing_duplicates_task(ti, data_file):
    logging.info("Starting missing and duplicates check task")
    df = pd.read_csv(data_file)
    missing_rows, duplicate_rows, status = find_missing_and_duplicates(df)
    update_results_xcom(ti, 'missing_duplicates', missing_rows + duplicate_rows, status)
    logging.info("Missing and duplicates check completed with status: %s", status)

def privacy_compliance_task(ti, data_file):
    logging.info("Starting privacy compliance task")
    df = pd.read_csv(data_file)
    rows, status = check_data_privacy(df)
    update_results_xcom(ti, 'privacy_compliance', rows, status)
    logging.info("Privacy compliance check completed with status: %s", status)

def emoji_detection_task(ti, data_file):
    logging.info("Starting emoji detection task")
    df = pd.read_csv(data_file)
    rows, status = detect_emoji(df)
    update_results_xcom(ti, 'emoji_detection', rows, status)
    logging.info("Emoji detection completed with status: %s", status)

def anomaly_detection_task(ti, data_file):
    logging.info("Starting anomaly detection task")
    df = pd.read_csv(data_file)
    status = detect_anomalies(df)
    update_results_xcom(ti, 'anomaly_detection', None, status)
    logging.info("Anomaly detection completed with status: %s", status)

def special_characters_detection_task(ti, data_file):
    logging.info("Starting special characters detection task")
    df = pd.read_csv(data_file)
    rows = check_only_special_characters(df)
    status = False if rows else True
    update_results_xcom(ti, 'special_characters_detection', rows, status)
    logging.info("Special characters detection completed with status: %s", status)

def review_length_checker_task(ti, data_file):
    logging.info("Starting review length check task")
    df = pd.read_csv(data_file)
    short_reviews, long_reviews, short_titles, long_titles, \
    short_reviews_flag, long_reviews_flag, short_titles_flag, long_titles_flag = check_review_title_length(df)
    # Store results in XCom individually
    update_results_xcom(ti, 'short_reviews', short_reviews, short_reviews_flag)
    update_results_xcom(ti, 'long_reviews', long_reviews, long_reviews_flag)
    update_results_xcom(ti, 'short_titles', short_titles, short_titles_flag)
    update_results_xcom(ti, 'long_titles', long_titles, long_titles_flag)
    logging.info("Review length check completed.")

def save_results(ti, **kwargs):
    """Collect and save results from XCom to a CSV file."""
    triggering_dag_id = kwargs.get('dag_run').conf.get('triggering_dag_id')
    if triggering_dag_id == 'sampling_train_dag':
        output_dir = "/opt/airflow/data/validation/training"
    elif triggering_dag_id == 'sampling_serve_dag':
        output_dir = "/opt/airflow/data/validation/serving"
    else:
        raise ValueError(f"Unknown triggering DAG ID: {triggering_dag_id}")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory for results at {output_dir}")

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
                "row_indices": str(task_result.get("row_indices", "")),
                "status": task_result.get("status", "")
            })
        else:
            logging.warning(f"No result found for component: {component}")
    # Process other tasks
    for task in other_task_names:
        task_result = ti.xcom_pull(key=f"{task}_results", task_ids=task)
        if task_result:
            results.append({
                "function": task,
                "row_indices": str(task_result.get("row_indices", "")),
                "status": task_result.get("status", "")
            })
        else:
            logging.warning(f"No result found for task: {task}")
    # Convert the list of results to a DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "validation_results.csv")
    results_df.to_csv(output_path, index=False)
    logging.info(f"Validation results saved successfully to {output_path}")

# Define the DAG
with DAG(
    dag_id='03_data_validation_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='DAG to perform data validation checks',
) as dag:

    set_input_path = PythonOperator(
        task_id='set_data_file',
        python_callable=set_data_file,
        provide_context=True,
    )

    schema_validation = PythonOperator(
        task_id='schema_validation',
        python_callable=lambda ti, **kwargs: schema_validation_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    range_check = PythonOperator(
        task_id='range_check',
        python_callable=lambda ti, **kwargs: range_check_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    missing_duplicates = PythonOperator(
        task_id='missing_duplicates',
        python_callable=lambda ti, **kwargs: missing_duplicates_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    privacy_compliance = PythonOperator(
        task_id='privacy_compliance',
        python_callable=lambda ti, **kwargs: privacy_compliance_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    emoji_detection = PythonOperator(
        task_id='emoji_detection',
        python_callable=lambda ti, **kwargs: emoji_detection_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    anomaly_detection = PythonOperator(
        task_id='anomaly_detection',
        python_callable=lambda ti, **kwargs: anomaly_detection_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    special_characters_detection = PythonOperator(
        task_id='special_characters_detection',
        python_callable=lambda ti, **kwargs: special_characters_detection_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    review_length_checker = PythonOperator(
        task_id='review_length_checker',
        python_callable=lambda ti, **kwargs: review_length_checker_task(ti, ti.xcom_pull(key='data_file_path')),
    )

    save_results_task = PythonOperator(
        task_id='save_results',
        python_callable=save_results,
        provide_context=True,
    )

    trigger_preprocessing_dag = TriggerDagRunOperator(
        task_id='trigger_preprocessing_data_pipeline',
        trigger_dag_id='04_data_preprocessing_dag',
        conf={"mode": "{{ dag_run.conf.get('triggering_dag_id', 'training') }}"},
    )


    set_input_path >> [schema_validation, range_check, missing_duplicates,
                       privacy_compliance, emoji_detection, anomaly_detection,
                       special_characters_detection, review_length_checker] >> save_results_task >> trigger_preprocessing_dag
