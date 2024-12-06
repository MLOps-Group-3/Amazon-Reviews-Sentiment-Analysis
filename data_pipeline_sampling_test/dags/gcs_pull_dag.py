from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from utils.data_collection.sampling_train import sample_training_data
from utils.data_collection.data_concat_train import concatenate_and_save_csv_files
from utils.data_collection.dynamic_month_train import get_next_training_period
from utils.data_collection.gcs_operations import pull_from_gcs, push_to_gcs
from utils.config import (
    CATEGORIES, 
    SAMPLED_TRAINING_DIRECTORY,
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY", "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME_MODEL", "model-deployment-from-airflow")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "amazonreviewssentimentanalysis")
GCS_REGION = os.getenv("GCS_REGION", "us-central1")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': 'vallimeenaavellaiyan@gmail.com',
}

# Define the DAG
with DAG(
    dag_id='02_GCS_pull_tasks',
    default_args=default_args,
    description='DAG to Pull Data From GCS Buckets',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    # Pull from GCS train sampling task
    pull_from_gcs_task = PythonOperator(
        task_id='pull_from_gcs_group_train_sampling',
        python_callable=pull_from_gcs,
        op_kwargs={
            'bucket_name': GCS_BUCKET_NAME,
            'source_blob_prefix': 'data/sampled/training/',
            'local_directory': SAMPLED_TRAINING_DIRECTORY,
            'service_account_key': GCS_SERVICE_ACCOUNT_KEY
        },
    )

    
    # Define the overall DAG structure
    pull_from_gcs_task
