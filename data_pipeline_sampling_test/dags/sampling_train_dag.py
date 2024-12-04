from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from utils.data_collection.sampling_train import sample_training_data
from utils.data_collection.data_concat_serve import concatenate_and_save_csv_files
from utils.data_collection.dynamic_month_train import get_next_training_period
from utils.data_collection.gcs_operations import push_to_gcs
from utils.config import CATEGORIES, SAMPLED_TRAINING_DIRECTORY
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY", "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME_MODEL", "model-deployment-from-airflow")

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
    dag_id='03_sampling_train_dag',
    default_args=default_args,
    description='DAG to sample training data dynamically',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    

    # Create tasks for each category
    category_tasks = []
    for category_name in CATEGORIES:
        # Get the next training period
        training_start_date, training_end_date = get_next_training_period(SAMPLED_TRAINING_DIRECTORY)
        task = PythonOperator(
            task_id=f'sample_training_{category_name}',
            python_callable=sample_training_data,
            op_kwargs={
                'category_name': category_name,
                'start_date': training_start_date,
                'end_date': training_end_date,
            },
        )
        category_tasks.append(task)

    # Create a task to concatenate data after all categories are sampled
    concat_task = PythonOperator(
        task_id='concatenate_training_data',
        python_callable=concatenate_and_save_csv_files,
        op_kwargs={
            'input_dir': SAMPLED_TRAINING_DIRECTORY,
            'output_file': f'{SAMPLED_TRAINING_DIRECTORY}/concatenated_training_data_{training_start_date}_to_{training_end_date}.csv',
        },
    )

    # Create a task to push the concatenated data to GCS
    push_to_gcs_task = PythonOperator(
        task_id='push_to_gcs',
        python_callable=push_to_gcs,
        op_kwargs={
            'bucket_name': GCS_BUCKET_NAME,
            'source_directory': SAMPLED_TRAINING_DIRECTORY,
            'destination_blob_prefix': 'data/sampled/training/',
            'service_account_key': GCS_SERVICE_ACCOUNT_KEY,
        },
    )

    # Trigger data validation DAG after the GCS push
    trigger_validation_dag = TriggerDagRunOperator(
        task_id='trigger_validation_dag',
        trigger_dag_id='03_data_validation_dag',
        wait_for_completion=False,
    )

    # Set up sequential dependencies
    if category_tasks:
        for i in range(len(category_tasks) - 1):
            category_tasks[i] >> category_tasks[i + 1]
        category_tasks[-1] >> concat_task >> push_to_gcs_task >> trigger_validation_dag
    else:
        concat_task >> push_to_gcs_task >> trigger_validation_dag