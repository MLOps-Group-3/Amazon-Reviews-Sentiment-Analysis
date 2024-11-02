import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start overall timing
overall_start_time = time.time()

# Download necessary NLTK packages and measure time
logging.info("Downloading required NLTK packages...")
start_time = time.time()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
logging.info(f"NLTK downloads completed in {time.time() - start_time:.2f} seconds.")

def get_synonyms(word):
    """
    Generate a set of synonyms for a given word using WordNet.

    Args:
        word (str): The word to find synonyms for.

    Returns:
        set: A set of synonyms for the word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

# Define aspects and keywords with relevant synonyms
logging.info("Defining aspects and keywords...")
start_time = time.time()
aspects = {
    "delivery": get_synonyms("delivery") | {"arrive", "shipping"},
    "quality": get_synonyms("quality") | {"craftsmanship", "durable"},
    "customer_service": get_synonyms("service") | {"support", "helpful", "response"},
    "product_design": get_synonyms("design") | {"appearance", "look", "style"},
    "cost": get_synonyms("cost") | get_synonyms("price") | {"value", "expensive", "cheap", "affordable", "$"}
}
logging.info(f"Aspects definition completed in {time.time() - start_time:.2f} seconds.")

def tag_and_expand_aspects_chunk(chunk, aspects):
    rows = []
    for idx, row in chunk.iterrows():
        review_text = row['text']
        user_id = row.get('user_id', 'N/A')
        asin = row.get('asin', 'N/A')
        
        sentences = sent_tokenize(review_text)
        aspect_sentences = {aspect: [] for aspect in aspects.keys()}
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            
            for aspect, keywords in aspects.items():
                if any(word in keywords for word, tag in pos_tags if tag.startswith(('NN', 'JJ'))):
                    aspect_sentences[aspect].append(sentence)
        
        for aspect, sentences in aspect_sentences.items():
            if sentences:
                row_copy = row.copy()
                row_copy['aspect'] = aspect
                row_copy['relevant_sentences'] = ' '.join(sentences)
                rows.append(row_copy)
    
    return pd.DataFrame(rows)

def parallel_process_aspects(df, aspects, num_workers=os.cpu_count()):
    chunk_size = len(df) // num_workers
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    results = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(tag_and_expand_aspects_chunk, chunk, aspects): chunk for chunk in chunks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logging.info("Completed a chunk of processing.")
    logging.info(f"Parallel processing completed in {time.time() - start_time:.2f} seconds.")
    
    return pd.concat(results, ignore_index=True)

if __name__ == "__main__":
    # Define the path to the data file
    file_path = "milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/cleaned_data_2018_2019.csv"

    # Load data and measure time
    logging.info(f"Loading data from {file_path}...")
    start_time = time.time()
    df_cleaned = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully in {time.time() - start_time:.2f} seconds.")
    
    # Process data in parallel for aspect tagging
    logging.info("Processing data for aspect tagging in parallel...")
    start_time = time.time()
    tagged_reviews_expanded = parallel_process_aspects(df_cleaned, aspects)
    logging.info(f"Aspect tagging and expansion completed in {time.time() - start_time:.2f} seconds.")
    
    # Display the resulting counts of aspects
    logging.info("Displaying aspect counts and DataFrame shape")
    print(tagged_reviews_expanded['aspect'].value_counts())
    print(tagged_reviews_expanded.shape)

    # Save the processed data to a CSV file and measure time
    output_path = 'milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/aspect_based.csv'
    start_time = time.time()
    tagged_reviews_expanded.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to '{output_path}' in {time.time() - start_time:.2f} seconds.")

    # Log overall script execution time
    logging.info(f"Total script execution time: {time.time() - overall_start_time:.2f} seconds.")
