from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import subprocess
import os

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
with DAG(
    'mlflow_tracking_dag',
    default_args=default_args,
    description='Run MLflow experiment tracking with experiment_runner.py',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
) as dag:

    def run_experiment_script():
        # Define the path to the script
        script_path = '/opt/airflow/src/experiment_runner.py'
        
        # Set the tracking URI to ensure it points to the MLflow server
        os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'

        # Run the script as a subprocess
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Script failed with error: {result.stderr}")
        print(result.stdout)

    # Create a PythonOperator to run the experiment script
    run_experiment = PythonOperator(
        task_id='run_mlflow_experiment',
        python_callable=run_experiment_script
    )

    run_experiment
