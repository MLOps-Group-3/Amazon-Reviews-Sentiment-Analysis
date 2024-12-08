from google.cloud import bigquery
import pandas as pd
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_id = os.getenv("GCS_PROJECT_ID")
dataset_id = os.getenv("GCS_DATASET_ID")
table_id = os.getenv("GCS_TABLE_ID")
output_path = "/Users/vallimeenaa/Downloads/Group 3 MLOps Project/3/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/Amazon_Sentiment_Analysis.csv"

# Path to your service account JSON file

def fetch_data_from_bigquery_and_save(project_id: str, dataset_id: str, table_id: str, output_path: str):
    """
    Fetches data from a BigQuery table and saves it to a local data folder as a CSV.

    :param project_id: GCP Project ID
    :param dataset_id: BigQuery Dataset ID
    :param table_id: BigQuery Table ID
    :param output_path: Path to save the output CSV
    """
    try:
        service_account_key_path = os.getenv("GCS_SERVICE_ACCOUNT_KEY")

        # Initialize the BigQuery client with the service account key file
        client = bigquery.Client.from_service_account_json(service_account_key_path)

        # Construct the BigQuery table reference
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        # Run the query to fetch all data from the table
        query = f"SELECT * FROM `{table_ref}`"
        logger.info(f"Executing query: {query}")
        query_job = client.query(query)

        # Convert query result to a pandas DataFrame
        df = query_job.to_dataframe()
        logger.info(f"Fetched {len(df)} rows from BigQuery table {table_ref}.")

        # Save DataFrame to the specified output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}.")

    except Exception as e:
        logger.error(f"Error fetching data from BigQuery: {e}")
        raise

if __name__ == "__main__":
    fetch_data_from_bigquery_and_save(project_id, dataset_id, table_id, output_path)
