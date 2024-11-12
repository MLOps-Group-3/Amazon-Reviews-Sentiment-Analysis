import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.data_collection.sampling import preprocess_file, sample_category, setup_dask_client
from utils.data_collection.data_concat import concatenate_and_save_csv_files
from utils.config import CATEGORIES, GCS_RAW_DATA_PATH, GCS_SAMPLED_DATA_PATH

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 30),
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': False,
    'email': 'vallimeenaavellaiyan@gmail.com',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    '02_data_sampling_dag',
    default_args=default_args,
    description='A DAG for preprocessing and sampling data using Dask',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
)

def preprocess_category(category_name):
    reviews_file = f"{GCS_RAW_DATA_PATH}/{category_name}_reviews.jsonl.gz"
    meta_file = f"{GCS_RAW_DATA_PATH}/{category_name}_meta.jsonl.gz"
    preprocess_file(reviews_file, GCS_SAMPLED_DATA_PATH)
    preprocess_file(meta_file, GCS_SAMPLED_DATA_PATH)

def run_sample_category(category_name):
    client = setup_dask_client()
    try:
        result = sample_category(category_name)
    finally:
        client.close()
    return result

with dag:
    with TaskGroup(group_id='preprocess_files') as preprocess_group:
        preprocess_tasks = [
            PythonOperator(
                task_id=f'preprocess_{category}',
                python_callable=preprocess_category,
                op_kwargs={'category_name': category},
            ) for category in CATEGORIES
        ]

    sample_tasks = []
    for category in CATEGORIES:
        sample_task = PythonOperator(
            task_id=f'sample_{category}',
            python_callable=run_sample_category,
            op_kwargs={'category_name': category},
        )
        sample_tasks.append(sample_task)

    concat_task = PythonOperator(
        task_id='concatenate_data',
        python_callable=concatenate_and_save_csv_files,
    )

    trigger_validation_dag = TriggerDagRunOperator(
        task_id='trigger_data_validation_dag',
        trigger_dag_id='03_data_validation_dag',
        wait_for_completion=False,
    )

    preprocess_group >> sample_tasks[0]
    for i in range(len(sample_tasks) - 1):
        sample_tasks[i] >> sample_tasks[i + 1]
    sample_tasks[-1] >> concat_task >> trigger_validation_dag