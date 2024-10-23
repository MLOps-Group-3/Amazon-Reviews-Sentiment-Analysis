from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
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

# Define the DAG
with DAG(
    dag_id='data_validation_dag',
    default_args=default_args,
    schedule_interval='@daily',
    description='DAG to perform data validation checks',
) as dag:

    # Task 1: Schema Validation
    schema_validation_task = PythonOperator(
        task_id='schema_validation',
        python_callable=validate_schema,
    )

    # Task 2: Range Check
    range_check_task = PythonOperator(
        task_id='range_check',
        python_callable=check_range,
    )

    # Task 3: Missing Values and Duplicates Check
    missing_duplicates_task = PythonOperator(
        task_id='missing_duplicates_check',
        python_callable=find_missing_and_duplicates,
    )

    # Task 4: Privacy Compliance Check
    privacy_compliance_task = PythonOperator(
        task_id='privacy_compliance_check',
        python_callable=check_data_privacy,
    )

    # Task to aggregate results (Runs only if all validations succeed)
    final_task = PythonOperator(
        task_id='final_task',
        python_callable=lambda: print("All validation checks passed successfully!"),
        trigger_rule='all_success',  # Ensures it runs only if all previous tasks succeed
    )

    # Run all validation tasks in parallel, then join to the final task
    [schema_validation_task, range_check_task, 
     missing_duplicates_task, privacy_compliance_task] >> final_task
