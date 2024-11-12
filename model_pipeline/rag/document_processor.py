import openai
import logging
import os
from dotenv import load_dotenv
from utils import load_prompt_template, split_text

logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()
# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE_PATH = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/prompt_template.txt"

def generate_prompt(text, previous_summary="None"):
    try:
        prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
        prompt = prompt_template.replace("{PREVIOUS_SUMMARY}", previous_summary).replace("{TEXT}", text)
        return prompt
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise

def analyze_chunk_with_summary(text, previous_summary="None"):
    if not openai.api_key:
        logger.error("OpenAI API key not found. Please set it as an environment variable.")
        raise ValueError("OpenAI API key not found. Set it as an environment variable.")

    prompt = generate_prompt(text, previous_summary)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please provide responses in plain JSON format without any code block markers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        logger.info("Chunk analyzed successfully.")
        
        # Return the raw output as a string
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise

def process_document_with_summary(document):
    text = document.get("text", "")
    chunks = split_text(text)
    previous_summary = "None"
    all_summaries = []

    for idx, chunk in enumerate(chunks):
        try:
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)}.")
            current_summary = analyze_chunk_with_summary(chunk, previous_summary)
            all_summaries.append(current_summary)
            previous_summary = current_summary
        except Exception as e:
            logger.error(f"Error processing chunk {idx + 1}: {e}")
            continue

    # Combine all summaries as a single string or list of summaries
    final_summary = "\n".join(all_summaries) if all_summaries else "No summary generated."
    document["analysis"] = final_summary
    logger.info("Document processed successfully.")
    return document
