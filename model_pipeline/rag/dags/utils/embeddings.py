import os
import logging
import json
import tempfile
from google.cloud import storage
from pinecone import Pinecone, ServerlessSpec
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_data_from_gcs(bucket_name, prefix):
    """
    Fetch JSON data from GCS and return a list of records suitable for embedding.
    """
    try:
        logging.info("Starting to fetch data from GCS...")
        service_account_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
        if not service_account_path:
            raise ValueError("GCS_SERVICE_ACCOUNT_KEY not found in environment variables.")
        
        # Initialize GCS client
        logging.info("Initializing Google Cloud Storage client.")
        storage_client = storage.Client.from_service_account_json(service_account_path)
        bucket = storage_client.get_bucket(bucket_name)
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
        logging.error(f"Error fetching data from GCS: {e}")
        return []


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
        
        # Initialize Pinecone client
        logging.info("Initializing Pinecone client.")
        pinecone = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        
        if index_name not in pinecone.list_indexes():
            logging.info(f"Creating a new Pinecone index: {index_name}")
            # Specify serverless spec
            pinecone.create_index(name=index_name, dimension=1536, spec=ServerlessSpec())
        
        index = pinecone.Index(index_name)
        
        # Generate embeddings and upsert
        vectors = []
        for record in records:
            text = record.get("text", "")
            metadata = {
                "year": record.get("year"),
                "month": record.get("month"),
                "category": record.get("category"),
                "average_rating": record.get("average_rating"),
                "dominant_sentiment": record.get("dominant_sentiment")
            }
            try:
                embedding = get_embedding(text, model="text-embedding-ada-002")
                vectors.append({"id": record["id"], "values": embedding, "metadata": metadata})
                logging.info(f"Generated embedding for record ID: {record['id']}")
            except Exception as e:
                logging.error(f"Error generating embedding for record ID {record.get('id', 'unknown')}: {e}")
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
    index_name = "amazon-reviews-index"
    
    # Fetch data from GCS
    records = fetch_data_from_gcs(bucket_name, prefix)
    
    # Upsert to Pinecone
    if records:
        upsert_to_pinecone(records, index_name)
    else:
        logging.warning("No valid records fetched from GCS. Skipping upsertion.")
