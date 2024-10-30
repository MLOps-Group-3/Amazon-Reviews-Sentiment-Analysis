from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

# Import utility functions
from utils.data_cleaning_pandas import clean_amazon_reviews
from utils.data_labeling import apply_labelling

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 30),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# File paths
data_file = "/opt/airflow/data/sampled_data_2018_2019.csv"
cleaned_data_file = "/opt/airflow/data/airflow/cleaned_data.csv"

# Define tasks
def data_cleaning_task():
    """Task to perform data cleaning."""
    df = pd.read_csv(data_file)
    df_cleaned = clean_amazon_reviews(df)
    df_cleaned.to_csv(cleaned_data_file, index=False)

def data_labeling_task():
    """Task to perform data labeling."""
    df = pd.read_csv(cleaned_data_file)
    df_labeled = apply_labelling(df)
    df_labeled.to_csv("/opt/airflow/data/airflow/labeled_data.csv", index=False)

# Define the DAG
with DAG(
    dag_id='data_pipeline_dag',
    default_args=default_args,
    schedule_interval='@daily',
    description='DAG for data cleaning and labeling',
) as dag:

    # Task 1: Data Cleaning
    data_cleaning = PythonOperator(
        task_id='data_cleaning',
        python_callable=data_cleaning_task,
    )

    # Task 2: Data Labeling
    data_labeling = PythonOperator(
        task_id='data_labeling',
        python_callable=data_labeling_task,
    )

    # Define task dependencies
    data_cleaning >> data_labeling
