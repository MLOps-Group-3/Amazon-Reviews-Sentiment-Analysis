from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from data_sampling import data_sampling
from config import CATEGORIES

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'amazon_reviews_data_sampling',
    default_args=default_args,
    description='A DAG for sampling Amazon review data',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

# Create a task for each category
sampling_tasks = []
for category in CATEGORIES:
    task = PythonOperator(
        task_id=f'sample_data_{category}',
        python_callable=data_sampling,
        op_kwargs={'category_name': category},
        dag=dag,
    )
    sampling_tasks.append(task)

# Create a dummy task to join sampling tasks
join_sampling_tasks = DummyOperator(
    task_id='join_sampling_tasks',
    dag=dag,
)

send_success_email = EmailOperator(
    task_id='send_success_email',
    to='subraamanian.ni@northeastern.edu',
    subject='Data Sampling Tasks Completed Successfully',
    html_content='The data sampling tasks for Amazon reviews have been completed successfully.',
    dag=dag,
)

send_failure_email = EmailOperator(
    task_id='send_failure_email',
    to='subraamanian.ni@northeastern.edu',
    subject='Data Sampling Tasks Failed',
    html_content='One or more data sampling tasks for Amazon reviews have failed.',
    dag=dag,
    trigger_rule='one_failed'
)

# Set up task dependencies
sampling_tasks >> join_sampling_tasks >> [send_success_email, send_failure_email]