# # from airflow import DAG
# # from airflow.operators.python import PythonOperator
# # from airflow.operators.email import EmailOperator
# # from airflow.operators.dummy import DummyOperator
# # from datetime import datetime, timedelta
# # from data_sampling import data_sampling
# # from config import CATEGORIES

# # default_args = {
# #     'owner': 'airflow',
# #     'depends_on_past': False,
# #     'start_date': datetime(2024, 10, 30),
# #     'email_on_failure': False,
# #     'email_on_retry': False,
# #     'retries': 1,
# #     'retry_delay': timedelta(minutes=5),
# # }

# # dag = DAG(
# #     'amazon_reviews_data_sampling',
# #     default_args=default_args,
# #     description='A DAG for sampling Amazon review data',
# #     schedule_interval=timedelta(days=1),
# #     catchup=False,
# #     max_active_runs=1,
# # )

# # # Create a task for each category
# # sampling_tasks = []
# # for category in CATEGORIES:
# #     task = PythonOperator(
# #         task_id=f'sample_data_{category}',
# #         python_callable=data_sampling,
# #         op_kwargs={'category_name': category},
# #         dag=dag,
# #     )
# #     sampling_tasks.append(task)

# # # Create a dummy task to join sampling tasks
# # join_sampling_tasks = DummyOperator(
# #     task_id='join_sampling_tasks',
# #     dag=dag,
# # )

# # # send_success_email = EmailOperator(
# # #     task_id='send_success_email',
# # #     to='subraamanian.ni@northeastern.edu',
# # #     subject='Data Sampling Tasks Completed Successfully',
# # #     html_content='The data sampling tasks for Amazon reviews have been completed successfully.',
# # #     dag=dag,
# # # )

# # # send_failure_email = EmailOperator(
# # #     task_id='send_failure_email',
# # #     to='subraamanian.ni@northeastern.edu',
# # #     subject='Data Sampling Tasks Failed',
# # #     html_content='One or more data sampling tasks for Amazon reviews have failed.',
# # #     dag=dag,
# # #     trigger_rule='one_failed'
# # # )

# # # Set up task dependencies
# # sampling_tasks >> join_sampling_tasks #>> [send_success_email, send_failure_email]

# from airflow import DAG
# from airflow.operators.dummy import DummyOperator
# from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
# from datetime import datetime, timedelta
# from config import CATEGORIES

# # Default arguments for the DAG
# default_args = {
#     'owner': 'airflow',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 10, 30),
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# # Define the DAG
# dag = DAG(
#     'amazon_reviews_data_sampling',
#     default_args=default_args,
#     description='A DAG for sampling Amazon review data using Spark',
#     schedule_interval=timedelta(days=1),
#     catchup=False,
#     max_active_runs=1,
# )

# # Create a task for each category using SparkSubmitOperator
# sampling_tasks = []
# for category in CATEGORIES:
#     task = SparkSubmitOperator(
#         task_id=f'sample_data_{category}',
#         conn_id='spark-conn',  # Spark connection ID in Airflow
#         application=f'/opt/airflow/dags/data_sampling.py',  # Path to your Spark job file
#         # name=f"sample_data_{category}",
#         conf={
#             "spark.master": "spark://spark-master:7077",
#             "spark.driver.extraJavaOptions": "-Djava.net.preferIPv4Stack=true",
#             "spark.executor.extraJavaOptions": "-Djava.net.preferIPv4Stack=true",
#         },
#         # executor_memory="2g",
#         # executor_cores=2,

#         application_args=[category],
#         dag=dag,
#     )
#     sampling_tasks.append(task)

# # Dummy task to join sampling tasks
# join_sampling_tasks = DummyOperator(
#     task_id='join_sampling_tasks',
#     dag=dag,
# )

# # Define the task sequence
# sampling_tasks >> join_sampling_tasks

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
from config import CATEGORIES

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'amazon_reviews_data_sampling',
    default_args=default_args,
    description='A DAG for sampling Amazon review data using Spark',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

# Initialize the first task as None
previous_task = None

# Create a task for each category in a serial sequence
for category in CATEGORIES:
    task = SparkSubmitOperator(
        task_id=f'sample_data_{category}',
        conn_id='spark-conn',  # Spark connection ID in Airflow
        application='/opt/airflow/dags/data_sampling.py',  # Path to your Spark job file
        conf={
            "spark.master": "spark://spark-master:7077",
            "spark.driver.extraJavaOptions": "-Djava.net.preferIPv4Stack=true",
            "spark.executor.extraJavaOptions": "-Djava.net.preferIPv4Stack=true",
        },
        application_args=[category],
        dag=dag,
    )
    
    # Set the previous task to depend on the current task to ensure serial execution
    if previous_task:
        previous_task >> task
    
    # Update the previous_task to the current one
    previous_task = task

# Dummy task to end the sequence
end_task = DummyOperator(
    task_id='join_sampling_tasks',
    dag=dag,
)

# Link the last sampling task to the end task
previous_task >> end_task
