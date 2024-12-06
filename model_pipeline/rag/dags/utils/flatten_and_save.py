import os
import json
import logging
from google.cloud import storage
from dotenv import load_dotenv


#load_env
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# GCS and Pinecone setup
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
PREFIX = "RAG/"
SERVICE_ACCOUNT_PATH = os.getenv("GCS_SERVICE_ACCOUNT_KEY")

# Initialize GCS client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def flatten_and_save_to_gcs(input_file: str):
    """
    Flatten and save JSON data into a hierarchical directory structure in GCS.

    Args:
        input_file (str): Path to the input JSON file.
    """
    try:
        # Load the input JSON file
        with open(input_file, "r") as file:
            data = json.load(file)

        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError("Input JSON should be a list of records.")

        # Iterate through each record in the list
        for record in data:
            process_record_to_gcs(record)

        logging.info("Flattening and uploading to GCS completed successfully.")
    except Exception as e:
        logging.error("Error during flattening and saving to GCS: %s", str(e), exc_info=True)
        raise


def process_record_to_gcs(record: dict):
    """
    Process a single record and upload its chunks to GCS.

    Args:
        record (dict): A single record from the JSON data.
    """
    try:
        # Extract metadata
        category = record.get("category", "Unknown_Category").strip()
        year = str(record.get("year", "Unknown_Year"))
        month = str(record.get("month", "Unknown_Month"))

        # Construct GCS path prefix
        base_gcs_path = f"{PREFIX}{category}/{year}/{month}/"

        # Process analysis data
        analysis = record.get("analysis", {})
        product_summaries = analysis.get("product_summaries", {})

        for subcategory, aspects in product_summaries.items():
            # Define subcategory file path in GCS
            file_name = f"{subcategory.replace(' ', '_')}.json"
            gcs_path = f"{base_gcs_path}{file_name}"

            # Convert data to JSON string
            json_data = json.dumps(aspects, indent=4)

            # Upload to GCS
            blob = bucket.blob(gcs_path)
            blob.upload_from_string(json_data, content_type="application/json")

            logging.info(f"Uploaded subcategory '{subcategory}' to GCS path: {gcs_path}")

    except Exception as e:
        logging.error("Error processing record for GCS: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    input_file = "/opt/airflow/data/refined_processed_documents.json"  # Replace with your JSON file path

    # Run the flatten and save process to GCS
    flatten_and_save_to_gcs(input_file)
