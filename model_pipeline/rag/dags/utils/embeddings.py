import os
from google.cloud import storage
from pinecone import Pinecone
from openai.embeddings_utils import get_embedding
import json
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def fetch_data_from_gcs(bucket_name, prefix):
    """
    Fetch JSON data from GCS and return a list of records.
    
    :param bucket_name: GCS bucket name.
    :param prefix: Prefix path in the GCS bucket.
    :return: List of records from GCS JSON files.
    """
    try:
        # Fetch GCS service account key path from environment
        service_account_path = os.getenv("GCS_SERVICE_ACCOUNT_KEY")
        if not service_account_path:
            raise ValueError("GCS_SERVICE_ACCOUNT_KEY not found in environment variables.")

        # Initialize GCS client
        storage_client = storage.Client.from_service_account_json(service_account_path)
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        records = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                print(f"Fetching file: {blob.name}")
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    temp_file_path = temp_file.name
                with open(temp_file_path, 'r') as f:
                    data = json.load(f)
                    records.extend(data)
                os.remove(temp_file_path)

        print(f"Fetched {len(records)} records from GCS.")
        return records
    except Exception as e:
        print(f"Error fetching data from GCS: {e}")
        return []


def upsert_to_pinecone(records, index_name):
    """
    Generate embeddings and upsert records to Pinecone.
    
    :param records: List of records to upsert.
    :param index_name: Name of the Pinecone index.
    """
    try:
        # Fetch Pinecone credentials from environment
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        if not pinecone_api_key or not pinecone_env:
            raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in environment variables.")

        # Initialize Pinecone
        pinecone = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=1536)
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
            except Exception as e:
                print(f"Error generating embedding for record {record['id']}: {e}")
                continue

        if vectors:
            try:
                index.upsert(vectors=vectors)
                print(f"Upserted {len(vectors)} vectors to Pinecone.")
            except Exception as e:
                print(f"Error during upsertion: {e}")

    except Exception as e:
        print(f"Error initializing or upserting to Pinecone: {e}")


# Example Usage
if __name__ == "__main__":
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
