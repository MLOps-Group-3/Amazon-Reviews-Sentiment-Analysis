import os
import json
import hashlib
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key, pool_threads=30)

# Define the Pinecone index name
index_name = "amazonsentimentanalysis"
dimension = 1536  # Ensure this matches your embedding size

# Check if the index exists or create a new one
if index_name not in [idx["name"] for idx in pc.list_indexes().get("indexes", [])]:
    print(f"Index '{index_name}' does not exist. Creating a new index...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Index '{index_name}' created successfully.")

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

# Function to prepare data from JSON files
def prepare_data_from_file(file_path):
    """Prepare data from a single JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        metadata = data.get("metadata")
        embedding = data.get("embedding")

        if not metadata or not embedding:
            print(f"Skipping {file_path}: Missing 'metadata' or 'embedding'.")
            return None

        if len(embedding) != dimension:
            print(f"Skipping {file_path}: Embedding size mismatch.")
            return None

        # Generate a unique ID using hashlib
        unique_id = hashlib.sha256(
            f"{metadata.get('subcategory')}_{metadata.get('year')}_{metadata.get('month')}".encode()
        ).hexdigest()
        

        return {"id": unique_id, "values": embedding, "metadata": metadata}
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Folder containing your JSON files
json_folder_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/embedding_meta"

# Process files in batches and upsert
def upsert_in_batches(folder_path, batch_size=100):
    """Upsert data in batches directly to Pinecone."""
    batch = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            
            # Prepare data for upsert
            data = prepare_data_from_file(file_path)
            if data:
                batch.append(data)
            
            # Upsert when batch size is reached
            if len(batch) >= batch_size:
                try:
                    index.upsert(vectors=batch)
                    print(f"Successfully upserted batch of size {len(batch)}.")
                except Exception as e:
                    print(f"Error during batch upsertion: {e}")
                finally:
                    batch = []  # Clear the batch

    # Upsert any remaining data
    if batch:
        try:
            index.upsert(vectors=batch)
            print(f"Successfully upserted final batch of size {len(batch)}.")
        except Exception as e:
            print(f"Error during final batch upsertion: {e}")

# Run the upsert process in batches
upsert_in_batches(json_folder_path, batch_size=100)

print("Finished processing all files.")

