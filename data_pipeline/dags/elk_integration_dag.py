from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'elk_integration',
    default_args=default_args,
    description='Integrate ELK stack with Amazon Reviews data',
    schedule_interval=timedelta(days=1),
)

run_logstash = BashOperator(
    task_id='run_logstash',
    bash_command='logstash -f /Users/prabhatchanda/PycharmProjects/Amazon-Reviews-Sentiment-Analysis/data_pipeline/elk/logstash/amazon_reviews.conf',
    dag=dag,
)

run_logstash