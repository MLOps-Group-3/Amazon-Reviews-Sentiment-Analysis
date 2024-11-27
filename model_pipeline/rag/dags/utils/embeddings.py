import os
import logging
import json
import tempfile
from google.cloud import storage
from pinecone import Pinecone
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper function to generate unique IDs
def generate_unique_id(category, year, month, dominant_sentiment):
    id_string = f"{category}-{year}-{month}-{dominant_sentiment}"
    return hashlib.sha256(id_string.encode()).hexdigest()

# Fetch data from GCS bucket
# Helper function to generate unique IDs
def generate_unique_id(category, year, month, dominant_sentiment):
    id_string = f"{category}-{year}-{month}-{dominant_sentiment}"
    return hashlib.sha256(id_string.encode()).hexdigest()

# Fetch data from GCS bucket
def fetch_data_from_gcs(bucket_name, prefix):
    """
    Fetch JSON data from GCS and return a list of records suitable for embedding.
    """
    try:
        logging.info("Starting to fetch data from GCS...")
        service_account_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        
        # Initialize GCS client
        logging.info("Initializing Google Cloud Storage client.")
        storage_client = storage.Client.from_service_account_json(service_account_path)
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        records = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                logging.info(f"Fetching file: {blob.name}")
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    temp_file_path = temp_file.name
                
                try:
                    with open(temp_file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            filtered_data = [item for item in data if isinstance(item, dict)]
                            records.extend(filtered_data)
                            logging.info(f"Processed {len(filtered_data)} dictionary items from {blob.name}.")
                        elif isinstance(data, dict):
                            records.append(data)
                            logging.info(f"Processed a single dictionary item from {blob.name}.")
                        else:
                            logging.warning(f"Skipping non-dictionary data in file {blob.name}: {type(data)}")
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON in file {blob.name}: {e}")
                finally:
                    os.remove(temp_file_path)

        logging.info(f"Finished fetching data from GCS. Total records fetched: {len(records)}")
        return records
    except Exception as e:
        logging.error(f"Error fetching data from GCS: {e}", exc_info=True)
        logging.error(f"Error fetching data from GCS: {e}", exc_info=True)
        return []

# Upsert data into Pinecone
# Upsert data into Pinecone
import os
import logging
import json
import tempfile
from google.cloud import storage
from pinecone import Pinecone
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Helper function to generate unique IDs
def generate_unique_id(category, year, month, dominant_sentiment):
    id_string = f"{category}-{year}-{month}-{dominant_sentiment}"
    return hashlib.sha256(id_string.encode()).hexdigest()

# Fetch data from GCS bucket
# Helper function to generate unique IDs
def generate_unique_id(category, year, month, dominant_sentiment):
    id_string = f"{category}-{year}-{month}-{dominant_sentiment}"
    return hashlib.sha256(id_string.encode()).hexdigest()

# Fetch data from GCS bucket
def fetch_data_from_gcs(bucket_name, prefix):
    """
    Fetch JSON data from GCS and return a list of records suitable for embedding.
    """
    try:
        logging.info("Starting to fetch data from GCS...")
        service_account_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        
        # Initialize GCS client
        logging.info("Initializing Google Cloud Storage client.")
        storage_client = storage.Client.from_service_account_json(service_account_path)
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        records = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                logging.info(f"Fetching file: {blob.name}")
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    temp_file_path = temp_file.name
                
                try:
                    with open(temp_file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            filtered_data = [item for item in data if isinstance(item, dict)]
                            records.extend(filtered_data)
                            logging.info(f"Processed {len(filtered_data)} dictionary items from {blob.name}.")
                        elif isinstance(data, dict):
                            records.append(data)
                            logging.info(f"Processed a single dictionary item from {blob.name}.")
                        else:
                            logging.warning(f"Skipping non-dictionary data in file {blob.name}: {type(data)}")
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON in file {blob.name}: {e}")
                finally:
                    os.remove(temp_file_path)

        logging.info(f"Finished fetching data from GCS. Total records fetched: {len(records)}")
        return records
    except Exception as e:
        logging.error(f"Error fetching data from GCS: {e}", exc_info=True)
        logging.error(f"Error fetching data from GCS: {e}", exc_info=True)
        return []

# Upsert data into Pinecone
# Upsert data into Pinecone
def upsert_to_pinecone(records, index_name):
    """
    Generate embeddings and upsert records to Pinecone.
    """
    try:
        logging.info("Starting the upsertion process to Pinecone...")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        if not pinecone_api_key or not pinecone_env:
            raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment variables.")
        
        # Initialize Pinecone client with environment
        logging.info("Initializing Pinecone client.")
        pinecone_client = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        
        # Check if the index exists
        existing_indexes_response = pinecone_client.list_indexes()
        existing_indexes = [idx['name'] for idx in existing_indexes_response.get('indexes', [])]
        if index_name not in existing_indexes:
            logging.error(f"Pinecone index '{index_name}' does not exist. Please create the index first.")
            return  # Exit if the index does not exist
        
        index = pinecone_client.Index(index_name)
        
        # Generate embeddings and upsert
        vectors = []
        for record in records:
            text = record.get("text", "")
            if not text:
                logging.warning(f"Record missing 'text' field. Skipping.")
                continue
            
            metadata = {
                "year": record.get("year"),
                "month": record.get("month"),
                "category": record.get("category"),
                "average_rating": record.get("average_rating"),
                "dominant_sentiment": record.get("dominant_sentiment"),
            }
            unique_id = generate_unique_id(
                metadata["category"], metadata["year"], metadata["month"], metadata["dominant_sentiment"]
            )
            
            try:
                # Check if the ID exists to avoid duplicate upserts
                existing_vector = index.fetch(ids=[unique_id])
                if existing_vector and existing_vector.get("vectors"):
                    logging.info(f"ID '{unique_id}' already exists in Pinecone. Skipping.")
                    continue

                # Generate embedding
                embedding = get_embedding(text, model="text-embedding-ada-002")
                vectors.append({"id": unique_id, "values": embedding, "metadata": metadata})
                logging.info(f"Generated embedding for record ID: {unique_id}")
            except Exception as e:
                logging.error(f"Error generating embedding for record ID {unique_id}: {e}")
                logging.error(f"Error generating embedding for record ID {unique_id}: {e}")
                continue
        
        if vectors:
            try:
                logging.info(f"Upserting {len(vectors)} vectors to Pinecone index: {index_name}")
                index.upsert(vectors=vectors)
                logging.info(f"Successfully upserted {len(vectors)} vectors to Pinecone.")
            except Exception as e:
                logging.error(f"Error during upsertion to Pinecone: {e}")
    except Exception as e:
        logging.error(f"Error initializing or upserting to Pinecone: {e}")


if __name__ == "__main__":
    logging.info("Starting the embedding and upsertion pipeline...")
    
    # GCS Configuration
    bucket_name = "amazon-reviews-sentiment-analysis"
    prefix = "RAG/"  # Directory or folder prefix in GCS bucket
    
    # Pinecone Configuration
    index_name = "amazonsentimentanalysis"
    
    # Fetch data from GCS
    records = fetch_data_from_gcs(bucket_name, prefix)
    
    # Upsert to Pinecone
    if records:
        upsert_to_pinecone(records, index_name)
    else:
        logging.warning("No valid records fetched from GCS. Skipping upsertion.")
