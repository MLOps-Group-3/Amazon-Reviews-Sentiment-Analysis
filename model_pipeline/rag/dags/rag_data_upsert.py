from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add the 'utils' folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import the flatten and save function and embeddings to Pinecone function from the utils folder
from flatten_and_save import flatten_and_save_to_gcs
from embeddings_to_pinecone import main as upsert_embeddings_to_pinecone

# Set up the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 11, 30),  # Adjust start date as needed
}

# Define the DAG
dag = DAG(
    'flatten_and_embeddings_dag',
    default_args=default_args,
    description='A DAG that flattens JSON, uploads to GCS, generates embeddings and upserts to Pinecone',
    schedule_interval=None,  # Set the schedule as per your needs (None for manual trigger)
    catchup=False,
)

# Define the flatten and save task
flatten_and_save_task = PythonOperator(
    task_id='flatten_and_save_to_gcs',
    python_callable=flatten_and_save_to_gcs,
    op_args=['/opt/airflow/data/refined_processed_documents.json'],  # Adjust input file path
    dag=dag,
)

# Define the embeddings and upsert task
upsert_embeddings_task = PythonOperator(
    task_id='upsert_embeddings_to_pinecone',
    python_callable=upsert_embeddings_to_pinecone,
    dag=dag,
)

# Set task dependencies
flatten_and_save_task >> upsert_embeddings_task  # Ensures flatten and save runs before embeddings upsert
