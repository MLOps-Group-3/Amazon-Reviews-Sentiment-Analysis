from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

# Import validation functions
from utils.schema_validation import validate_schema
from utils.range_checker import check_range
from utils.missing_duplicates_checker import find_missing_and_duplicates
from utils.privacy_compliance import check_data_privacy

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# File path of the data to validate
data_file = "/opt/airflow/data/sampled_data_2018_2019.csv"

def schema_validation_task():
    """Task to perform schema validation."""
    df = pd.read_csv(data_file)
    validate_schema(df)

def range_check_task():
    """Task to perform range check."""
    df = pd.read_csv(data_file)
    check_range(df)

def missing_duplicates_task():
    """Task to check for missing values and duplicates."""
    df = pd.read_csv(data_file)
    find_missing_and_duplicates(df)

def privacy_compliance_task():
    """Task to check for privacy compliance."""
    df = pd.read_csv(data_file)
    check_data_privacy(df)

# Define the DAG
with DAG(
    dag_id='data_validation_dag',
    default_args=default_args,
    schedule_interval='@daily',
    description='DAG to perform data validation checks',
) as dag:

    # Task 1: Schema Validation (Linear)
    schema_validation = PythonOperator(
        task_id='schema_validation',
        python_callable=schema_validation_task,
    )

    # Task 2: Range Check (Parallel)
    range_check = PythonOperator(
        task_id='range_check',
        python_callable=range_check_task,
    )

    # Task 3: Missing Values and Duplicates Check (Parallel)
    missing_duplicates = PythonOperator(
        task_id='missing_duplicates_check',
        python_callable=missing_duplicates_task,
    )

    # Task 4: Privacy Compliance Check (Parallel)
    privacy_compliance = PythonOperator(
        task_id='privacy_compliance_check',
        python_callable=privacy_compliance_task,
    )

    # Task 5: Final Task (Aggregate Results)
    final_task = PythonOperator(
        task_id='final_task',
        python_callable=lambda: print("All validation checks passed successfully!"),
        trigger_rule='all_success',
    )

    # Define dependencies
    schema_validation >> [range_check, missing_duplicates, privacy_compliance] >> final_task
