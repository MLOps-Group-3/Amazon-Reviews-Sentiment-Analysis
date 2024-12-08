import os
import json
import logging
from collections import defaultdict
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to your service account JSON file
SERVICE_ACCOUNT_KEY_PATH = os.getenv("GCS_SERVICE_ACCOUNT_KEY_LOCAL")

# GCS bucket and folder details
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
RAG_FOLDER = "RAG"

# Output file
OUTPUT_PATH = "/Users/vallimeenaa/Downloads/Group 3 MLOps Project/3/Amazon-Reviews-Sentiment-Analysis/model_pipeline/Streamlit/hierarchical_metadata.json"

def extract_metadata_from_gcs(bucket_name, folder_name):
    """Extract metadata from GCS bucket by iterating over the folder structure."""
    # Authenticate with GCS
    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY_PATH)
    bucket = storage_client.bucket(bucket_name)

    # Initialize metadata map
    metadata_map = defaultdict(lambda: defaultdict(set))

    # Iterate through blobs in the GCS bucket
    blobs = storage_client.list_blobs(bucket_name, prefix=folder_name + "/")
    for blob in blobs:
        # Skip if it's the folder itself
        if blob.name.endswith("/"):
            continue

        # Parse the blob name to extract category, year, and month
        parts = blob.name[len(folder_name) + 1:].split("/")
        if len(parts) < 3:
            logging.warning(f"Unexpected blob path format: {blob.name}")
            continue

        category, year, month_file = parts[0], parts[1], parts[2]
        month = os.path.splitext(month_file)[0]  # Strip file extension to get the month

        if category and year and month:
            metadata_map[category][year].add(month)
        else:
            logging.warning(f"Missing metadata in blob path: {blob.name}")

    # Convert sets to lists for JSON serialization
    json_serializable_map = {
        category: {year: list(months) for year, months in years.items()}
        for category, years in metadata_map.items()
    }

    return json_serializable_map

def save_hierarchical_metadata(metadata_map, output_path):
    """Save hierarchical metadata to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metadata_map, f, indent=4)

    logging.info(f"Hierarchical metadata saved to {output_path}")

# Main script logic
if __name__ == "__main__":
    try:
        # Extract metadata from the GCS bucket
        metadata_map = extract_metadata_from_gcs(BUCKET_NAME, RAG_FOLDER)

        # Save the hierarchical metadata to a file
        save_hierarchical_metadata(metadata_map, OUTPUT_PATH)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
