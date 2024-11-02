from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
from utils.data_collection.sampling import process_category
from airflow.models import Variable
from utils.data_collection.config import CATEGORIES

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
    'sampling',
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

# Set task dependencies if needed
# For example, if you want tasks to run in sequence:
for i in range(1, len(category_tasks)):
    category_tasks[i-1] >> category_tasks[i]

# Alternatively, if you want tasks to run in parallel, you don't need to set dependencies
