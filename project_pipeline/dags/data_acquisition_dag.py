from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from data_utils.data_collection.data_acquisition import acquire_data
from airflow.models import Variable

# Set the LOG_DIRECTORY variable
Variable.set("LOG_DIRECTORY", "/opt/airflow/logs", description="Directory for storing logs")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),  # Start from January 1, 2024
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': False,
    'email': 'vallimeenaavellaiyan@gmail.com'
}

# Define the DAG with a single cron expression that runs on both:
# - 1st of every month
# - 2nd of March, July, and November
dag = DAG(
    '01_data_collection_dag',
    default_args=default_args,
    description='A DAG for acquiring Amazon review data',
    schedule_interval='0 0 1,2 * *',  # Runs on the 1st and 2nd of every month
    catchup=False,
    max_active_runs=1,
)

def determine_acquisition_type(**kwargs):
    """
    Determines if the run is for monthly or quarterly acquisition.
    - Monthly: Runs on the 1st of every month.
    - Quarterly: Runs on the 2nd of March, July, and November.
    
    Returns:
        str: Either "monthly" or "quarterly".
        Raises an exception if the date is invalid.
    """
    execution_date = kwargs['execution_date']
    
    if execution_date.day == 1:
        return 'monthly'
    elif execution_date.day == 2 and execution_date.month in [3, 7, 11]:
        return 'quarterly'
    else:
        raise ValueError(f"Invalid execution date: {execution_date}")

# Task to determine acquisition type (monthly or quarterly)
acquisition_type_task = PythonOperator(
    task_id='determine_acquisition_type',
    python_callable=determine_acquisition_type,
    provide_context=True,
    dag=dag,
)

# Task to acquire data (common for both monthly and quarterly runs)
acquire_data_task = PythonOperator(
    task_id='acquire_data',
    python_callable=acquire_data,
    dag=dag,
)

# Trigger the GCS pull DAG with acquisition type as configuration
trigger_gcs_pull_dag = TriggerDagRunOperator(
    task_id='trigger_gcs_pull_dag',
    trigger_dag_id='02_GCS_pull_tasks',  # ID of the next DAG to trigger
    conf={'acquisition_type': "{{ task_instance.xcom_pull(task_ids='determine_acquisition_type') }}"},
    wait_for_completion=False,
    dag=dag,
)

# Define task dependencies
acquisition_type_task >> acquire_data_task >> trigger_gcs_pull_dag
