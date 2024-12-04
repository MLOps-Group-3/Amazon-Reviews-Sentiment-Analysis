import os
import json
import logging
import hashlib
from google.cloud import storage
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import tempfile


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")

# GCS and Pinecone setup
BUCKET_NAME = "amazon-reviews-sentiment-analysis"
PREFIX = "RAG/"
SERVICE_ACCOUNT_PATH = "/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/amazonreviewssentimentanalysis-8dfde6e21c1d.json"

# Pinecone Initialization
pc = Pinecone(api_key=api_key, pool_threads=30)
index_name = "amazonsentimentanalysis"
dimension = 1536  # Ensure this matches your embedding size

# Check if the index exists or create a new one
if index_name not in [idx["name"] for idx in pc.list_indexes().get("indexes", [])]:
    logging.info(f"Index '{index_name}' does not exist. Creating a new index...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logging.info(f"Index '{index_name}' created successfully.")

# Get the host for the index
index_host = None
for idx in pc.list_indexes().get("indexes", []):
    if idx["name"] == index_name:
        index_host = idx["host"]
        break

if not index_host:
    raise ValueError(f"Host not found for index '{index_name}'.")

# Use the Index class to interact with the index
index = pc.Index(host=index_host)

# Fetch data from GCS bucket
def fetch_data_from_gcs(bucket_name, prefix):
    try:
        logging.info("Starting to fetch data from GCS...")
        storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_PATH)
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        records = []
        file_paths = []
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
                            file_paths.extend([blob.name] * len(filtered_data))
                        elif isinstance(data, dict):
                            records.append(data)
                            file_paths.append(blob.name)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON in file {blob.name}: {e}")
                finally:
                    os.remove(temp_file_path)
        logging.info(f"Finished fetching data from GCS. Total records fetched: {len(records)}")
        return records, file_paths
    except Exception as e:
        logging.error(f"Error fetching data from GCS: {e}", exc_info=True)
        return [], []

def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def extract_metadata(file_path):
    parts = file_path.split('/')
    if len(parts) < 5:
        logging.warning(f"Invalid file path structure: {file_path}")
        return None
    category = parts[1]
    subcategory = parts[4]
    year = parts[2]
    month = parts[3]
    return {'category': category, 'subcategory': subcategory, 'year': year, 'month': month}

def prepare_data_for_upsert(record, file_path):
    """
    Prepare a record for upsertion by combining metadata and content.
    
    Parameters:
        record (dict): Content of the file.
        file_path (str): Path of the file in GCS.
    
    Returns:
        dict: A record ready for Pinecone upsertion.
    """
    try:
        # Extract metadata from the file path
        metadata = extract_metadata(file_path)
        if not metadata:
            return None

        # Combine metadata and record content for embedding
        metadata_str = " ".join([f"{key}: {value}" for key, value in metadata.items()])
        text = metadata_str + " " + json.dumps(record)  # Combine metadata and text content

        # Generate embedding
        embedding = generate_embedding(text)
        if not embedding:
            return None

        # Create a unique ID for the record
        unique_id = hashlib.sha256(f"{metadata.get('subcategory')}_{metadata.get('year')}_{metadata.get('month')}".encode()).hexdigest()

        return {
            "id": unique_id,
            "values": embedding,  # Combined embedding
            "metadata": metadata  # Metadata as separate field
        }
    except Exception as e:
        logging.error(f"Error processing record: {e}")
        return None

def upsert_to_pinecone(records, file_paths):
    batch = []
    for record, file_path in zip(records, file_paths):
        logging.info(f"Processing record from file: {file_path}")
        data = prepare_data_for_upsert(record, file_path)
        if data:
            batch.append(data)
        if len(batch) >= 100:  # Batch size limit for upsert
            try:
                index.upsert(vectors=batch)
                logging.info(f"Successfully upserted batch of size {len(batch)}.")
            except Exception as e:
                logging.error(f"Error during batch upsertion: {e}")
            finally:
                batch = []  # Reset the batch for the next set of records
    if batch:
        try:
            index.upsert(vectors=batch)
            logging.info(f"Successfully upserted final batch of size {len(batch)}.")
        except Exception as e:
            logging.error(f"Error during final batch upsertion: {e}")

def main():
    logging.info("Starting the embedding generation and Pinecone upsertion pipeline...")
    
    # Fetch data from GCS
    records, file_paths = fetch_data_from_gcs(BUCKET_NAME, PREFIX)
    if not records:
        logging.warning("No records fetched from GCS. Exiting.")
        return
    
    # Directly upsert the embeddings to Pinecone
    upsert_to_pinecone(records, file_paths)
    
    logging.info(f"Pipeline completed successfully.")

if __name__ == "__main__":
    main()
