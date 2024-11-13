import logging
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math
import os

class HierarchicalSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", max_chunk_tokens=400, final_summary_length=150):
        """
        Initializes the summarizer with a specified model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_chunk_tokens = max_chunk_tokens  # Maximum tokens per chunk
        self.final_summary_length = final_summary_length  # Target length for the final summary
        logging.info(f"Initialized summarizer with model {model_name}")

    def chunk_text(self, text):
        """
        Splits the text into chunks based on max_chunk_tokens.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids'][0]

        # Calculate the number of chunks
        num_chunks = math.ceil(len(input_ids) / self.max_chunk_tokens)
        
        # Split the input_ids into chunks
        chunks = [input_ids[i*self.max_chunk_tokens: (i+1)*self.max_chunk_tokens] for i in range(num_chunks)]
        
        # Decode each chunk back to text
        chunk_texts = [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        logging.info(f"Text split into {len(chunk_texts)} chunks.")
        return chunk_texts

    def summarize_chunk(self, text, max_length=150, min_length=30):
        """
        Summarizes a single chunk of text.
        """
        # Use truncation only in the tokenizer, not in the generate function
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.debug(f"Generated chunk summary: {summary}")
        return summary


    def hierarchical_summarize(self, text):
        """
        Applies hierarchical summarization by summarizing chunks, then summarizing the combined chunk summaries.
        """
        # Step 1: Split text into chunks
        chunks = self.chunk_text(text)
        logging.info(f"Total Chunks: {len(chunks)}")

        # Step 2: Summarize each chunk
        chunk_summaries = [self.summarize_chunk(chunk) for chunk in chunks]
        logging.info("First-level chunk summaries generated.")

        # Step 3: Combine chunk summaries into a single text
        combined_summary_text = " ".join(chunk_summaries)

        # Step 4: Summarize the combined summary text for the final summary
        final_summary = self.summarize_chunk(combined_summary_text, max_length=self.final_summary_length)
        logging.info("Final summary generated.")

        return final_summary, chunk_summaries

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Path to the JSON file
    file_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/document_store.json"
    output_file_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/rag/data/summary_output.json"
    
    # Load JSON data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Initialize the summarizer
    summarizer = HierarchicalSummarizer(model_name="facebook/bart-large-cnn")

    # Prepare output data structure
    summary_results = []

    # Loop over each entry in the JSON data
    for entry in data:
        text = entry.get("text", "")
        year = entry.get("year", "Unknown Year")
        month = entry.get("month", "Unknown Month")
        category = entry.get("category", "Unknown Category")
        
        # Generate the hierarchical summary for each entry
        final_summary, chunk_summaries = summarizer.hierarchical_summarize(text)
        
        # Log the results for each entry
        logging.info(f"Year: {year}, Month: {month}, Category: {category}")
        logging.info("Final Summary:\n" + final_summary)
        logging.info("Chunk Summaries:\n" + str(chunk_summaries))
        
        # Append the result to summary_results
        summary_results.append({
            "year": year,
            "month": month,
            "category": category,
            "final_summary": final_summary,
            "chunk_summaries": chunk_summaries
        })

    # Save all summaries to an output JSON file
    with open(output_file_path, "w") as output_file:
        json.dump(summary_results, output_file, indent=4)
    logging.info(f"Summarization results saved to {output_file_path}")
