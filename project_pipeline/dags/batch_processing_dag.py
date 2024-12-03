from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from batch_processing import read_and_preprocess_data, upload_to_gcs, create_bq_table, submit_batch_prediction, create_output_table
import os

# Set the path to your service account key
service_account_path = "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': 'vallimeenaavellaiyan@gmail.com'  # Global recipient for all tasks
}

dag = DAG(
    'batch_processing',
    default_args=default_args,
    description='Batch processing pipeline for Amazon Reviews Sentiment Analysis',
    schedule_interval=timedelta(days=1),
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
    dag=dag,
)

t3 = PythonOperator(
    task_id='create_bq_table',
    python_callable=create_bq_table,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='submit_batch_prediction',
    python_callable=submit_batch_prediction,
    provide_context=True,
    dag=dag,
)

t5 = PythonOperator(
    task_id='create_output_table',
    python_callable=create_output_table,
    provide_context=True,
    dag=dag,
)

t1 >> t2 >> t3 >> t4 >> t5
