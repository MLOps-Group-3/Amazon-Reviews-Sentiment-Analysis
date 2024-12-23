from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
from utils.data_collection.data_acquisition import acquire_data
from airflow.models import Variable

# Set the LOG_DIRECTORY variable
Variable.set("LOG_DIRECTORY", "/opt/airflow/logs", description="Directory for storing logs")

# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 10, 30),
#     'email_on_failure': True,
#     'email_on_retry': True,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5)
# }
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,         # Send email on failure
    'email_on_retry': False,           # Send email on retry
    'email_on_success': False,        # Optional: email on success
    'email': 'vallimeenaavellaiyan@gmail.com'  # Global recipient for all tasks
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
    dag=dag,
)

# send_success_email = EmailOperator(
#     task_id='send_success_email',
#     # to='subramanian.ni@northeastern.edu',
#     subject='Data Acquisition Task Completed Successfully',
#     html_content='The data acquisition task for Amazon reviews has been completed successfully.',
#     dag=dag,
# )

# send_failure_email = EmailOperator(
#     task_id='send_failure_email',
#     to='subraamanian.ni@northeastern.edu',
#     subject='Data Acquisition Task Failed',
#     html_content='The data acquisition task for Amazon reviews has failed.',
#     dag=dag,
#     trigger_rule='one_failed'
# )

trigger_sampling_dag = TriggerDagRunOperator(
    task_id='trigger_sampling_dag',
    trigger_dag_id='02_data_sampling_dag',  # ID of the next DAG to trigger
    wait_for_completion=False,  # Wait until sampling_dag completes
    dag=dag,
)


acquire_data_task >> trigger_sampling_dag
# >> [send_success_email, send_failure_email]
