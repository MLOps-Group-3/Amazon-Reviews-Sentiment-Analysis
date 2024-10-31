# Import necessary libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
import json

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

data_file = "/opt/airflow/data/sampled_data_2018_2019.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def update_results_xcom(ti, function_name, row_indices, status):
    """Push task results to XCom. JSON format nested structures for specific tasks."""
    
    # Flatten row indices and status for review_length_checker
    if function_name == 'review_title_length':
        row_indices_flat = json.dumps(row_indices)  # Convert nested structure to JSON string
        status_flat = json.dumps(status)  # Convert nested structure to JSON string
    else:
        row_indices_flat = row_indices
        status_flat = status

    # Push flattened or original results to XCom
    ti.xcom_push(key=f"{function_name}_results", value={
        'function': function_name,
        'row_indices': row_indices_flat,
        'status': status_flat
    })


def schema_validation_task(ti):
    logging.info("Starting schema validation task")
    df = pd.read_csv(data_file)
    status = validate_schema(df)
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

    # Since check_only_special_characters only returns one value, we update the code to match
    rows = check_only_special_characters(df)
    status = False if rows else True
    
    update_results_xcom(ti, 'special_characters_detection', rows, status)
    logging.info("Special characters detection completed with status: %s", status)


def review_length_checker_task(ti):
    logging.info("Starting review length check task")
    df = pd.read_csv(data_file)
    
    result = check_review_title_length(df)
    rows = {
        "short_reviews": result["short_reviews"],
        "long_reviews": result["long_reviews"],
        "short_titles": result["short_titles"],
        "long_titles": result["long_titles"]
    }
    status = {
        "short_reviews": result['status_flags']['short_reviews_flag'],
        "long_reviews": result['status_flags']['long_reviews_flag'],
        "short_titles": result['status_flags']['short_titles_flag'],
        "long_titles": result['status_flags']['long_titles_flag']
    }

    update_results_xcom(ti, 'review_title_length', rows, status)
    logging.info("Review length check completed with status flags: %s", status)

def save_results(ti):
    """Collect and save results from XCom to a CSV file."""
    task_names = [
        'schema_validation', 'range_check', 'missing_duplicates',
        'privacy_compliance', 'emoji_detection', 'anomaly_detection',
        'special_characters_detection', 'review_title_length'
    ]

    results = []
    for task in task_names:
        task_result = ti.xcom_pull(key=f"{task}_results", task_ids=task)
        if task_result:
            if task == 'review_title_length':
                # Load JSON strings back to dictionaries
                task_result['row_indices'] = json.loads(task_result['row_indices'])
                task_result['status'] = json.loads(task_result['status'])
            results.append(task_result)
        else:
            logging.warning(f"Result missing for task: {task}")

    # Convert list of results to DataFrame and save
    results_df = pd.json_normalize(results, sep='_')  # Unnest dictionary columns
    results_df.to_csv("/opt/airflow/data/validation_results.csv", index=False)
    logging.info("Results saved successfully to validation_results.csv")

# DAG definition remains the same

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
        trigger_rule='all_done',
    )

    parallel_tasks = [
        schema_validation, range_check, missing_duplicates,
        privacy_compliance, emoji_detection, anomaly_detection,
        special_characters_detection, review_length_checker
    ]
    
    load_data >> parallel_tasks >> final_task