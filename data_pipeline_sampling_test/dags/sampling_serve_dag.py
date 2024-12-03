from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from utils.data_collection.dynamic_month import get_next_serving_month
from utils.config import CATEGORIES, SAMPLED_SERVING_DIRECTORY
from utils.data_collection.sampling_serve import sample_serving_data
from utils.data_collection.data_concat_serve import concatenate_and_save_csv_files
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
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
    schedule_interval=None,  # Run on the 1st day of every month
    catchup=False,
    max_active_runs=1,
) as dag:

    # Calculate dynamic year and month
    year, next_month = get_next_serving_month(SAMPLED_SERVING_DIRECTORY)

    # Create tasks for each category
    category_tasks = []
    for category_name in CATEGORIES:
        task = PythonOperator(
            task_id=f'sample_serving_{category_name}',
            python_callable=sample_serving_data,
            op_kwargs={
                'category_name': category_name,
                'year': year,
                'month': next_month,
            },
        )
        category_tasks.append(task)

    # Create a task to concatenate data after all categories are sampled
    concat_task = PythonOperator(
        task_id='concatenate_serving_data',
        python_callable=concatenate_and_save_csv_files,
        op_kwargs={
            'input_dir': SAMPLED_SERVING_DIRECTORY,
            'output_file': f"{SAMPLED_SERVING_DIRECTORY}/concatenated_serving_data_{year}_{str(next_month).zfill(2)}.csv",
        },
    )

    # Trigger data validation DAG after concatenation (optional, commented for now)
    # trigger_validation_dag = TriggerDagRunOperator(
    #     task_id='trigger_validation_dag',
    #     trigger_dag_id='03_data_validation_dag',  # Replace with your validation DAG ID
    #     wait_for_completion=False,
    # )

    # Set up sequential dependencies for category tasks
    for i in range(len(category_tasks) - 1):
        category_tasks[i] >> category_tasks[i + 1]

    # Connect the last category task to concatenation
    category_tasks[-1] >> concat_task

    # Optionally trigger the validation DAG after concatenation
    # concat_task >> trigger_validation_dag
