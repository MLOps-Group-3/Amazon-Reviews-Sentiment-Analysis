import os
import json
import logging
from pinecone.grpc import PineconeGRPC as Pinecone
from google.cloud import storage
from dotenv import load_dotenv
import openai
import urllib.parse

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")
gcp_bucket_name = os.getenv("GCS_BUCKET_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
service_key_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)
index_name = "amazonsentimentanalysis"
index_host = None

# Get the host for the index
for idx in pc.list_indexes().get("indexes", []):
    if idx["name"] == index_name:
        index_host = idx["host"]
        break

if not index_host:
    raise ValueError(f"Host not found for index '{index_name}'.")

index = pc.Index(host=index_host)

# Initialize GCP storage client with service account key
try:
    storage_client = storage.Client.from_service_account_json(service_key_path)
    logging.info("GCP storage client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing GCP storage client: {e}")
    raise

# Set up OpenAI API key
openai.api_key = openai_api_key

# Function to generate text embeddings using OpenAI's text-embedding-ada-002
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

# Function to query Pinecone
def query_pinecone(embedding, top_k=10):
    """Query Pinecone to get top-k related vectors."""
    try:
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.get("matches", [])
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        return []

# Function to fetch JSON data from GCP
def fetch_from_gcp(metadata):
    """Fetch the JSON file from GCP based on metadata."""
    try:
        if not gcp_bucket_name:
            logging.error("Bucket name is not set. Please check your environment variables.")
            return None

        # Get the GCP bucket
        bucket = storage_client.bucket(gcp_bucket_name)

        # Construct the file path without encoding special characters like `/`
        file_path = f"RAG/{metadata.get('category', 'unknown')}/{metadata.get('year', 'unknown')}/{metadata.get('month', 'unknown')}/{metadata.get('subcategory', 'unknown')}"
        
        # Log the file path for debugging
        logging.info(f"Fetching file: {file_path}")

        # Fetch the blob (file) from the bucket
        blob = bucket.blob(file_path)

        # Download the file as text and parse it into JSON
        data = blob.download_as_text()
        print(json.loads(data))
        return json.loads(data)
    except Exception as e:
        logging.error(f"Error fetching file from GCP: {e}")
        return None


def prepare_llm_input(results):
    """Prepare input for the LLM based on retrieval results."""
    summary_data = {}

    for match in results:
        metadata = match.get("metadata", {})
        data = fetch_from_gcp(metadata)
        if data:
            # Iterate through all aspects in the fetched data
            for aspect, details in data.items():
                summary_data[aspect] = {
                    "sentiment": details.get("sentiment", "Unknown"),
                    "summary": details.get("summary", "No summary available."),
                }

    if not summary_data:
        return (
            "I wasn’t able to retrieve any relevant information to provide a summary. "
            "It seems our database might have missed a spot! Please rest assured that we’re working hard to make things better. "
            "In the meantime, feel free to explore other categories or ask me about something else. 😊"
        )

    # Construct LLM input with structured paragraphs for a polished response
    llm_input = (
        "Here’s what I found based on your query. I’ve summarized the information into key areas for clarity:\n\n"
    )

    for aspect, details in summary_data.items():
        if details["sentiment"] == "Unknown":
            llm_input += (
                f"- **{aspect}**: Unfortunately, sentiment data for this aspect is currently unavailable. "
                "We’re on it and hope to have this gap filled soon. Thanks for your patience! 😊\n\n"
            )
        else:
            llm_input += (
                f"- **{aspect}**:\n"
                f"  Sentiment: {details['sentiment']}.\n"
                f"  In summary: {details['summary']}\n\n"
            )

    llm_input += (
        "If you’d like more specific details or a deeper dive into any of these areas, let me know. "
        "We’re constantly improving our system to provide better insights and more comprehensive responses. Thank you for using our service! 🚀"
    )

    return llm_input




# Function to get response from LLM
def get_llm_response(llm_input):
    """Generate a conversational response using OpenAI's GPT-4."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": llm_input}],
            max_tokens=200,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Error generating LLM response: {e}")
        return "An error occurred while generating a response."

# Main process for user input
def process_text_query(input_text, top_k=10):
    """Process a text query, retrieve relevant results, and generate a response."""
    logging.info("Generating embedding for input text...")
    embedding = generate_embedding(input_text)

    if not embedding:
        return "Failed to generate embedding for the input text."

    logging.info("Querying Pinecone...")
    matches = query_pinecone(embedding, top_k=top_k)

    if not matches:
        logging.warning("No matches found.")
        return "No relevant data found for your query."

    logging.info("Preparing input for LLM...")
    llm_input = prepare_llm_input(matches)

    logging.info("Generating response using LLM...")
    response = get_llm_response(llm_input)

    return response

# Example usage
if __name__ == "__main__":
    user_input = "How was ice makers durability in 2019?"
    response = process_text_query(user_input, top_k=2)
    print("Chatbot Response:")
    print(response)
