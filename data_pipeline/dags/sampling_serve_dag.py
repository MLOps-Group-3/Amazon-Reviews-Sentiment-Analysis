from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from utils.data_collection.sampling_serve import sample_serving_data
from utils.data_collection.data_concat import concatenate_and_save_csv_files
from utils.config import CATEGORIES

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),  # Replace with your desired start date
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': 'vallimeenaavellaiyan@gmail.com',  # Replace with your email
}

# Define the DAG
with DAG(
    dag_id='sampling_serve_dag',
    default_args=default_args,
    description='DAG to sample serving data dynamically on the 1st day of each month',
    schedule_interval='0 0 1 * *',  # Run on the 1st day of every month
    catchup=False,
    max_active_runs=1,
) as dag:

    # Create tasks for each category
    category_tasks = []
    for category_name in CATEGORIES:
        task = PythonOperator(
            task_id=f'sample_serving_{category_name}',
            python_callable=sample_serving_data,
            op_kwargs={
                'category_name': category_name,
                'year': '{{ macros.ds_format(ds, "%Y-%m-%d", "%Y") }}',  # Extract year from execution date
                'month': '{{ macros.ds_format(ds, "%Y-%m-%d", "%m") }}',  # Extract month from execution date
            },
        )
        category_tasks.append(task)

    # Create a task to concatenate data after all categories are sampled
    concat_task = PythonOperator(
        task_id='concatenate_serving_data',
        python_callable=concatenate_and_save_csv_files,
        op_kwargs={
            'input_dir': '/opt/airflow/data/sampled/serving',  # Directory for sampled serving files
            'output_file': '/opt/airflow/data/sampled/serving/concatenated_serving_data.csv',  # Output file path
        },
    )

    # Trigger data validation DAG after concatenation
    trigger_validation_dag = TriggerDagRunOperator(
        task_id='trigger_validation_dag',
        trigger_dag_id='03_data_validation_dag',  # ID of the validation DAG
        wait_for_completion=False,
    )

    # Set up sequential dependencies
    for i in range(len(category_tasks) - 1):
        category_tasks[i] >> category_tasks[i + 1]

    # Connect the last category task to concatenation and validation
    category_tasks[-1] >> concat_task >> trigger_validation_dag
