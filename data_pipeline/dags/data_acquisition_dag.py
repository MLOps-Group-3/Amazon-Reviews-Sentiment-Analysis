from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from datetime import datetime, timedelta

# Import the acquire_data function
from utils.data_collection.data_acquisition import acquire_data

# Set the GCS bucket and pipeline path
GCS_BUCKET = "amazon-reviews-sentiment-analysis"
GCS_PIPELINE_PATH = f"gs://{GCS_BUCKET}/pipeline"
GCS_LOG_DIRECTORY = f"{GCS_PIPELINE_PATH}/logs"
GCS_DATA_DIRECTORY = f"{GCS_PIPELINE_PATH}/data"

# Set the LOG_DIRECTORY variable
Variable.set("LOG_DIRECTORY", GCS_LOG_DIRECTORY, description="GCS directory for storing logs")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': False,
    'email': 'vallimeenaavellaiyan@gmail.com'
}

dag = DAG(
    '01_data_collection_dag',
    default_args=default_args,
    description='A DAG for acquiring Amazon review data',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
)

acquire_data_task = PythonOperator(
    task_id='acquire_data',
    python_callable=acquire_data,
    op_kwargs={
        'gcs_pipeline_path': GCS_PIPELINE_PATH,
        'gcs_data_directory': GCS_DATA_DIRECTORY
    },
    dag=dag,
)

trigger_sampling_dag = TriggerDagRunOperator(
    task_id='trigger_sampling_dag',
    trigger_dag_id='02_data_sampling_dag',
    wait_for_completion=False,
    dag=dag,
)

acquire_data_task >> trigger_sampling_dag
