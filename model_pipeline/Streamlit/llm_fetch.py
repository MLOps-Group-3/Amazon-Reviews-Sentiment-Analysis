import os
import json
import logging
from pinecone.grpc import PineconeGRPC as Pinecone
from google.cloud import storage
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")
gcp_bucket_name = os.getenv("GCS_BUCKET_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")
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


def generate_embedding(text):
    """Generate text embeddings using OpenAI's text-embedding-ada-002."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None


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


def filter_pinecone_matches(matches, category=None, subcategory=None, year=None, month=None):
    """
    Filter Pinecone matches based on the dropdown values for accuracy.

    Parameters:
        matches (list): List of Pinecone matches.
        category (str): Selected category filter.
        subcategory (str): Selected subcategory filter.
        year (str): Selected year filter.
        month (str): Selected month filter.

    Returns:
        list: Filtered matches.
    """
    filtered_matches = []
    for match in matches:
        metadata = match.get("metadata", {})
        if category and metadata.get("category") != category:
            continue
        if subcategory and metadata.get("subcategory") != subcategory:
            continue
        if year and str(metadata.get("year")) != str(year):
            continue
        if month and str(metadata.get("month")) != str(month):
            continue
        filtered_matches.append(match)
    return filtered_matches


def fetch_from_gcp(metadata):
    """Fetch the JSON file from GCP based on metadata."""
    try:
        if not gcp_bucket_name:
            logging.error("Bucket name is not set. Please check your environment variables.")
            return None

        bucket = storage_client.bucket(gcp_bucket_name)
        file_path = f"RAG/{metadata.get('category', 'unknown')}/{metadata.get('year', 'unknown')}/{metadata.get('month', 'unknown')}/{metadata.get('subcategory', 'unknown').split('.')[0]}.json"
        
        logging.info(f"Fetching file: {file_path}")
        blob = bucket.blob(file_path)

        data = blob.download_as_text()
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
        
        # Ensure data is a dictionary
        if isinstance(data, dict):
            for aspect, details in data.items():
                # Ensure details is a dictionary
                if isinstance(details, dict):
                    summary_data[aspect] = {
                        "sentiment": details.get("sentiment", "Unknown"),
                        "summary": details.get("summary", "No summary available."),
                    }
                else:
                    # Handle cases where details are not as expected
                    summary_data[aspect] = {
                        "sentiment": "Unknown",
                        "summary": "No details available.",
                    }

    if not summary_data:
        return (
            "According to our records, we couldn’t retrieve any relevant information for your query. "
            "We are continuously working to enhance our data coverage and will provide updates as soon as possible."
        )

    llm_input = (
        "Here’s what we found based on your query. These insights summarize the key aspects:\n\n"
    )

    for aspect, details in summary_data.items():
        if details["sentiment"] == "Unknown":
            llm_input += (
                f"- **{aspect}**: Sentiment data is currently unavailable. "
                "We are working on addressing this gap. Thank you for your patience.\n\n"
            )
        else:
            llm_input += (
                f"- **{aspect}**:\n"
                f"  Sentiment: {details['sentiment']}\n"
                f"  Summary: {details['summary']}\n\n"
            )

    llm_input += (
        "If further details or additional insights are required, please feel free to ask. "
        "Thank you for your continued support!"
    )

    return llm_input



def get_llm_response(llm_input, is_out_of_context=False):
    """
    Generate a clear, elaborative response using OpenAI's GPT-4, focusing on detailed explanations 
    for internal stakeholders based on provided data and summarizing it effectively.
    """
    try:
        # Define the chatbot's purpose and instructions
        system_message = (
            "You are a data analysis assistant for an internal product review team. "
            "Your task is to analyze and interpret sentiment analysis data provided as input, "
            "explain the findings comprehensively for stakeholders, and summarize the key insights concisely. "
            "Your audience is internal stakeholders, so you should focus on clarity, relevance, and professionalism. "
            "Avoid generic or vague responses. Use phrases like 'According to our analysis' or "
            "'The records indicate' to reinforce credibility and ensure the response is focused and actionable."
            "If data is missing or unavailable, acknowledge it professionally and suggest ongoing efforts to address the gap."
        )

        if is_out_of_context:
            out_of_context_response = (
                "It seems like your question is outside the scope of my expertise. My focus is on analyzing product reviews, "
                "sentiment insights, and related data. Feel free to ask me specific questions about these topics, and I’ll provide detailed insights!"
            )
            return out_of_context_response

        # Use the system message and user query for the LLM response
        response = openai.ChatCompletion.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": llm_input}
            ],
            max_tokens=700,
            temperature=0.7,
        )

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        logging.error(f"Error generating LLM response: {e}")
        return (
            "An error occurred while generating a response. Please try again later. If the issue persists, contact support."
        )

def process_text_query(input_text, top_k=10, category=None, subcategory=None, year=None, month=None):
    """
    Process a text query, retrieve relevant results, and generate a response.

    Parameters:
        input_text (str): User query.
        top_k (int): Number of top results to retrieve.
        category (str): Selected category filter.
        subcategory (str): Selected subcategory filter.
        year (str): Selected year filter.
        month (str): Selected month filter.

    Returns:
        str: Chatbot response.
    """
    logging.info("Processing user query...")

    embedding = generate_embedding(input_text)
    if not embedding:
        return "Failed to generate embedding for the input text."

    matches = query_pinecone(embedding, top_k=top_k)
    filtered_matches = filter_pinecone_matches(matches, category, subcategory, year, month)

    if not filtered_matches:
        return "No relevant data found for your query."

    llm_input = prepare_llm_input(filtered_matches)
    response = get_llm_response(llm_input)
    return response


if __name__ == "__main__":
    user_input = "How was coffee makers in 2019?"
    response = process_text_query(user_input, top_k=4)
    print("Chatbot Response:")
    print(response)
