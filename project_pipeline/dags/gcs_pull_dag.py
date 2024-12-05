from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from data_utils.data_collection.sampling_train import sample_training_data
from data_utils.data_collection.data_concat_train import concatenate_and_save_csv_files
from data_utils.data_collection.dynamic_month_train import get_next_training_period
from data_utils.data_collection.gcs_operations import pull_from_gcs, push_to_gcs
from data_utils.config import (
    CATEGORIES, 
    SAMPLED_TRAINING_DIRECTORY,
    SAMPLED_SERVING_DIRECTORY,
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY", "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME_MODEL", "model-deployment-from-airflow")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "amazonreviewssentimentanalysis")
GCS_REGION = os.getenv("GCS_REGION", "us-central1")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': 'vallimeenaavellaiyan@gmail.com',
}

with DAG(
    dag_id='02_GCS_pull_tasks',
    default_args=default_args,
    description='DAG to Pull Data From GCS Buckets',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    def choose_task(**context):
        acquisition_type = context['dag_run'].conf.get('acquisition_type')
        if acquisition_type == 'monthly':
            return 'pull_from_gcs_group_serving_sampling'
        elif acquisition_type == 'quarterly':
            return 'pull_from_gcs_group_training_sampling'
        else:
            raise ValueError(f"Invalid acquisition type: {acquisition_type}")

    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=choose_task,
        provide_context=True,
    )

    pull_from_gcs_serving_task = PythonOperator(
        task_id='pull_from_gcs_group_serving_sampling',
        python_callable=pull_from_gcs,
        op_kwargs={
            'bucket_name': GCS_BUCKET_NAME,
            'source_blob_prefix': 'data/sampled/serving/',
            'local_directory': SAMPLED_SERVING_DIRECTORY,
            'service_account_key': GCS_SERVICE_ACCOUNT_KEY
        },
    )

    pull_from_gcs_training_task = PythonOperator(
        task_id='pull_from_gcs_group_training_sampling',
        python_callable=pull_from_gcs,
        op_kwargs={
            'bucket_name': GCS_BUCKET_NAME,
            'source_blob_prefix': 'data/sampled/training/',
            'local_directory': SAMPLED_TRAINING_DIRECTORY,
            'service_account_key': GCS_SERVICE_ACCOUNT_KEY
        },
    )

    trigger_sampling_serve_dag = TriggerDagRunOperator(
        task_id='trigger_sampling_serve_dag',
        trigger_dag_id='03_sampling_serve_dag',
        wait_for_completion=False,
    )

    trigger_sampling_train_dag = TriggerDagRunOperator(
        task_id='trigger_sampling_train_dag',
        trigger_dag_id='03_sampling_train_dag',
        wait_for_completion=False,
    )

    # Define the task dependencies
    branch_task >> [pull_from_gcs_serving_task, pull_from_gcs_training_task]
    pull_from_gcs_serving_task >> trigger_sampling_serve_dag
    pull_from_gcs_training_task >> trigger_sampling_train_dag
