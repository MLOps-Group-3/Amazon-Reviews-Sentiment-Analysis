import pandas as pd
import logging
import os
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform.model_monitoring.objective import ObjectiveConfig, SkewDetectionConfig, DriftDetectionConfig
from google.cloud.aiplatform.model_monitoring import EmailAlertConfig

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

def upload_to_gcs(GCS_SERVICE_ACCOUNT_KEY, GCS_BUCKET_NAME, **kwargs):
    ti = kwargs['ti']
    try:
        logging.info("Starting upload_to_gcs function")
        
        # Set the path to your service account key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_KEY
        
        local_file_path = ti.xcom_pull(key='output_path', task_ids='read_and_preprocess_data')
        
        bucket_name = GCS_BUCKET_NAME
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

def create_bq_table(GCS_SERVICE_ACCOUNT_KEY, GCS_BUCKET_NAME, GCS_PROJECT_ID, **kwargs):
    try:
        logging.info("Starting create_bq_table function")
        
        # Set the path to your service account key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_KEY
        
        client = bigquery.Client()
        
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/batch_processing_input/Serving Batches/processed_batch_data.csv"
        project_id = GCS_PROJECT_ID
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

def submit_batch_prediction(GCS_SERVICE_ACCOUNT_KEY, GCS_PROJECT_ID, GCS_REGION, **kwargs):
    try:
        logging.info("Starting submit_batch_prediction function")
        
        # Set the path to your service account key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_KEY
        
        aiplatform.init(project=GCS_PROJECT_ID, location=GCS_REGION)
        
        model_name = f"projects/{GCS_PROJECT_ID}/locations/{GCS_REGION}/models/7777458171236319232"
        
        input_table = "bq://amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data"
        
        output_table_prefix = "bq://amazonreviewssentimentanalysis.amazon_reviews_sentiment.processed_batch_data_w_predictions"

        train_table = "bq://amazonreviewssentimentanalysis.batch_processing_input_batch_test_data.batch_processing_input_train"

        batch_prediction_name = "Amazon-Month-Batch-BQ-rearranged"
        
        model = aiplatform.Model(model_name=model_name)
        
        # batch_prediction_job = model.batch_predict(
        #     job_display_name=batch_prediction_name,
        #     bigquery_source=input_table,
        #     bigquery_destination_prefix=output_table_prefix,
        #     machine_type="n1-standard-2",
        #     starting_replica_count=2,
        #     max_replica_count=2,
        #     sync=True
        # )
        
        # output_info_table = batch_prediction_job.output_info.bigquery_output_table

        # Define the skew detection config
        skew_detection_config = SkewDetectionConfig(
            data_source=train_table,
            # target_field="price",
            skew_thresholds={
                "text": 0.3,
                "price_missing": 0.1,
                "helpful_vote": 0.2,
                "verified_purchase": 0.1,
                "price": 0.3
            }
        )

        # Define the drift detection config (They say in documentation this doesn't work)
        drift_detection_config = DriftDetectionConfig(
            drift_thresholds={
                "text": 0.3,
                "price_missing": 0.1,
                "helpful_vote": 0.2,
                "verified_purchase": 0.1
            }
        )

        # Create the objective config
        model_monitoring_objective_config = ObjectiveConfig(
            skew_detection_config=skew_detection_config,
            drift_detection_config=drift_detection_config
        )

        # Create the email alert configuration
        model_monitoring_alert_config = EmailAlertConfig(
            user_emails=["mlopsgrp3@gmail.com"]
        )

        # Create the batch prediction job
        batch_prediction_job = aiplatform.BatchPredictionJob.create(
            job_display_name=batch_prediction_name,
            model_name=model_name,
            instances_format="bigquery",
            predictions_format="bigquery",
            bigquery_source=input_table,
            bigquery_destination_prefix=output_table_prefix,
            model_parameters=None,
            machine_type="n1-standard-4",
            accelerator_type=None,
            accelerator_count=None,
            starting_replica_count=1,
            max_replica_count=10,
            generate_explanation=False,
            explanation_metadata=None,
            explanation_parameters=None,
            encryption_spec_key_name=None,
            sync=True,
            model_monitoring_objective_config=model_monitoring_objective_config,
            model_monitoring_alert_config=model_monitoring_alert_config
        )

        # Wait for the job to complete
        batch_prediction_job.wait()

        logging.info(f"Batch prediction job completed with state: {batch_prediction_job.state}")
        logging.info(f"Prediction results saved to: {batch_prediction_job.output_info.bigquery_output_dataset}")

        logging.info(f"Batch Prediction Job {batch_prediction_name} completed.")
        
    except Exception as e:
      logging.error(f"Error in submit_batch_prediction function: {str(e)}")
      raise

def create_output_table(GCS_SERVICE_ACCOUNT_KEY, **kwargs):
    try:
        logging.info("Starting create_output_table function")

        # Set the path to your service account key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_KEY

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
