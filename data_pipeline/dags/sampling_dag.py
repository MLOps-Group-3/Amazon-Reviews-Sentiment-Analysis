from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from airflow.models.baseoperator import chain
from datetime import datetime, timedelta
from utils.data_collection.sampling import sample_category
from utils.data_collection.data_concat import concatenate_and_save_csv_files
from utils.config import CATEGORIES

# Set the LOG_DIRECTORY variable
Variable.set("LOG_DIRECTORY", "/opt/airflow/logs", description="Directory for storing logs")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 30),
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': False,
    'email': 'vallimeenaavellaiyan@gmail.com',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    '02_data_sampling_pipeline',
    default_args=default_args,
    description='A DAG for sampling data',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
)

# Create tasks for each category
category_tasks = []
for category_name in CATEGORIES:
    task = PythonOperator(
        task_id=f'sample_{category_name}',
        python_callable=sample_category,
        op_kwargs={'category_name': category_name},
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
    wait_for_completion=False,  # Wait until sampling_dag completes
    dag=dag,
)

# Set up sequential task dependencies
chain(category_tasks + [concat_task, trigger_validation_dag])