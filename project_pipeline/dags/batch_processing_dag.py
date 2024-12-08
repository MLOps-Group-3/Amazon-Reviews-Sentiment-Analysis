from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from serve_utils.batch_processing import read_and_preprocess_data, upload_to_gcs, create_bq_table, submit_batch_prediction, create_output_table, drop_batch_pred_output_table
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
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 3),
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': 'vallimeenaavellaiyan@gmail.com'  # Global recipient for all tasks
}

dag = DAG(
    '06_batch_processing_dag',
    default_args=default_args,
    description='Batch processing pipeline for Amazon Reviews Sentiment Analysis',
    schedule_interval=None,
    catchup=False
)

t1 = PythonOperator(
    task_id='read_and_preprocess_data',
    python_callable=read_and_preprocess_data,
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='upload_to_gcs',
    python_callable=upload_to_gcs,
    provide_context=True,
    op_kwargs={
            "GCS_SERVICE_ACCOUNT_KEY": GCS_SERVICE_ACCOUNT_KEY,
            "GCS_BUCKET_NAME": GCS_BUCKET_NAME
        },
    dag=dag,
)

t3 = PythonOperator(
    task_id='create_bq_table',
    python_callable=create_bq_table,
    provide_context=True,
    op_kwargs={
            "GCS_SERVICE_ACCOUNT_KEY": GCS_SERVICE_ACCOUNT_KEY,
            "GCS_PROJECT_ID": GCS_PROJECT_ID,
            "GCS_BUCKET_NAME": GCS_BUCKET_NAME
        },
    dag=dag,
)

t4 = PythonOperator(
    task_id='drop_batch_pred_output_table',
    python_callable=drop_batch_pred_output_table,
    provide_context=True,
    op_kwargs={
            "GCS_SERVICE_ACCOUNT_KEY": GCS_SERVICE_ACCOUNT_KEY,
            "GCS_PROJECT_ID": GCS_PROJECT_ID,
            "GCS_REGION": GCS_REGION
        },
    dag=dag,
)

t5 = PythonOperator(
    task_id='submit_batch_prediction',
    python_callable=submit_batch_prediction,
    provide_context=True,
    op_kwargs={
            "GCS_SERVICE_ACCOUNT_KEY": GCS_SERVICE_ACCOUNT_KEY,
            "GCS_PROJECT_ID": GCS_PROJECT_ID,
            "GCS_REGION": GCS_REGION
        },
    dag=dag,
)

t6 = PythonOperator(
    task_id='create_output_table',
    python_callable=create_output_table,
    provide_context=True,
    op_kwargs={
            "GCS_SERVICE_ACCOUNT_KEY": GCS_SERVICE_ACCOUNT_KEY
        },
    dag=dag,
)

t1 >> t2 >> t3 >> t4 >> t5 >> t6
