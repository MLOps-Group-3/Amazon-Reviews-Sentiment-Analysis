from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
from utils.data_collection.sampling import process_category
from airflow.models import Variable
from utils.data_collection.data_concat import concatenate_and_save_csv_files
from utils.config import CATEGORIES

# Set the LOG_DIRECTORY variable
Variable.set("LOG_DIRECTORY", "/opt/airflow/logs", description="Directory for storing logs")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 30),
    'email_on_failure': True,
    'email_on_retry': True,
    'email_on_success': False,
    'email': 'vallimeenaavellaiyan@gmail.com',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    '02_data_sampling_pipeline',
    default_args=default_args,
    description='A DAG for sampling data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

# Create tasks for each category
category_tasks = []
for category in CATEGORIES:
    task = PythonOperator(
        task_id=f'process_{category}',
        python_callable=process_category,
        op_kwargs={'category': category},
        dag=dag,
    )
    category_tasks.append(task)

# Create a task to concatenate data
concat_task = PythonOperator(
    task_id='concatenate_data',
    python_callable=concatenate_and_save_csv_files,
    dag=dag,
)

# Trigger data validation dag after concatenation
trigger_validation_dag = TriggerDagRunOperator(
    task_id='trigger_data_validation_dag',
    trigger_dag_id='03_data_validation_pipeline',  # ID of the next DAG to trigger
    wait_for_completion=True,  # Wait until sampling_dag completes
    dag=dag,
)

# Set up task dependencies
for task in category_tasks:
    task >> concat_task

concat_task >> trigger_validation_dag