import pandas as pd
import logging
import os
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_and_preprocess_data(**kwargs):
    ti = kwargs['ti']
    try:
        logging.info("Starting read_and_preprocess_data function")
        file_path = "/opt/airflow/data/cleaned/cleaned_data.csv"
        logging.info(f"Reading data from {file_path}")
        data = pd.read_csv(file_path)
        
        logging.info("Preprocessing data")
        data['text'] = data['text'].fillna('')
        data['title'] = data['title'].fillna('')
        data['price'] = pd.to_numeric(data['price'].replace("unknown", None), errors='coerce')
        data['price_missing'] = data['price'].isna().astype(int)
        data['price'] = data['price'].fillna(0).astype(float)
        data['helpful_vote'] = data['helpful_vote'].fillna(0).astype(int)
        data['verified_purchase'] = data['verified_purchase'].apply(lambda x: True if x else False)
        
        logging.info("Reordering columns")
        first_columns = ["text", "price", "price_missing", "helpful_vote", "verified_purchase"]
        all_columns = data.columns.tolist()
        remaining_columns = [col for col in all_columns if col not in first_columns]
        new_column_order = first_columns + remaining_columns
        data = data[new_column_order]
        
        output_path = "/opt/airflow/data/batch_processing_input/processed_batch_data.csv"
        logging.info(f"Saving processed data to {output_path}")
        data.to_csv(output_path, index=False)
        
        ti.xcom_push(key='output_path', value=output_path)
        logging.info("read_and_preprocess_data function completed successfully")
    except Exception as e:
        logging.error(f"Error in read_and_preprocess_data function: {str(e)}")
        raise

def upload_to_gcs(**kwargs):
    ti = kwargs['ti']
    try:
        logging.info("Starting upload_to_gcs function")
        
        # Set the path to your service account key
        service_account_path = "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        
        local_file_path = ti.xcom_pull(key='output_path', task_ids='read_and_preprocess_data')
        
        bucket_name = "model-deployment-from-airflow"
        destination_blob_name = "batch_processing_input/Serving Batches/processed_batch_data.csv"
        
        logging.info(f"Uploading file {local_file_path} to GCS bucket {bucket_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        
        logging.info("upload_to_gcs function completed successfully")
    except Exception as e:
        logging.error(f"Error in upload_to_gcs function: {str(e)}")
        raise

def create_bq_table(**kwargs):
    try:
        logging.info("Starting create_bq_table function")
        
        # Set the path to your service account key
        service_account_path = "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        
        client = bigquery.Client()
        
        gcs_uri = "gs://model-deployment-from-airflow/batch_processing_input/Serving Batches/processed_batch_data.csv"
        project_id = "amazonreviewssentimentanalysis"
        dataset_id = "amazon_reviews_sentiment"
        table_id = "processed_batch_data"
        
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        logging.info(f"Loading data from {gcs_uri} to BigQuery table {table_ref}")
        
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        
        load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
        
        load_job.result()  # Wait for the job to complete
        
        logging.info("create_bq_table function completed successfully")
    except Exception as e:
        logging.error(f"Error in create_bq_table function: {str(e)}")
        raise

def submit_batch_prediction(**kwargs):
    try:
        logging.info("Starting submit_batch_prediction function")
        
        # Set the path to your service account key
        service_account_path = "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        
        aiplatform.init(project="amazonreviewssentimentanalysis", location="us-central1")
        
        model_name = "projects/amazonreviewssentimentanalysis/locations/us-central1/models/7777458171236319232"
        
        input_table = "bq://amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data"
        
        output_table_prefix = "bq://amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data_w_predictions"
        
        batch_prediction_name = "Amazon-Month-Batch-BQ-rearranged"
        
        model = aiplatform.Model(model_name=model_name)
        
        batch_prediction_job = model.batch_predict(
            job_display_name=batch_prediction_name,
            bigquery_source=input_table,
            bigquery_destination_prefix=output_table_prefix,
            machine_type="n1-standard-2",
            starting_replica_count=2,
            max_replica_count=2,
            sync=True
        )
        
        output_info_table = batch_prediction_job.output_info.bigquery_output_table
        
        logging.info(f"Batch Prediction Job {batch_prediction_name} completed.")
        
    except Exception as e:
      logging.error(f"Error in submit_batch_prediction function: {str(e)}")
      raise

def create_output_table(**kwargs):
    try:
        logging.info("Starting create_output_table function")

        # Set the path to your service account key
        service_account_path = "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

        client = bigquery.Client()

        query = """
        DROP TABLE IF EXISTS `amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data_w_predictions_clean`;
        CREATE TABLE `amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data_w_predictions_clean` AS
        SELECT * EXCEPT(prediction),
          JSON_EXTRACT_SCALAR(prediction, '$[0]') AS sentiment_label,
          CAST(JSON_EXTRACT_SCALAR(prediction, '$[1].POS') AS FLOAT64) AS POS_confidence,
          CAST(JSON_EXTRACT_SCALAR(prediction, '$[1].NEU') AS FLOAT64) AS NEU_confidence,
          CAST(JSON_EXTRACT_SCALAR(prediction, '$[1].NEG') AS FLOAT64) AS NEG_confidence
        FROM `amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data_w_predictions`;
        """

        logging.info("Executing SQL query to clean prediction results")

        job = client.query(query)
        job.result()  # Wait for the query to finish

        logging.info("create_output_table function completed successfully")
    except Exception as e:
        logging.error(f"Error in create_output_table function: {str(e)}")
        raise
