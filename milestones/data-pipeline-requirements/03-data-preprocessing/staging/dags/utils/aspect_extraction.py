import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Downloading required NLTK packages...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# nltk.data.path.append('/home/hrs/nltk_data')

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
    logging.debug(f"Synonyms for '{word}': {synonyms}")
    return synonyms

# Define aspects and keywords with relevant synonyms
logging.info("Defining aspects and keywords...")
aspects = {
    "delivery": get_synonyms("delivery") | {"arrive", "shipping"},
    "quality": get_synonyms("quality") | {"craftsmanship", "durable"},
    "customer_service": get_synonyms("service") | {"support", "helpful", "response"},
    "product_design": get_synonyms("design") | {"appearance", "look", "style"},
    "cost": get_synonyms("cost") | get_synonyms("price") | {"value", "expensive", "cheap", "affordable","$"}
}

def tag_and_expand_aspects(df, aspects):
    """
    Processes each review to identify sentences relevant to specified aspects.
    For each detected aspect, a new row is created with the relevant sentences.

    Args:
        df (pd.DataFrame): DataFrame containing the review text and associated metadata.
        aspects (dict): Dictionary of aspects with related keywords.

    Returns:
        pd.DataFrame: DataFrame with separate rows for each detected aspect, including relevant sentences.
    """
    rows = []
    logging.info("Starting aspect tagging and expansion...")
    for idx, row in df.iterrows():
        review_text = row['text']
        user_id = row.get('user_id', 'N/A')
        asin = row.get('asin', 'N/A')
        
        logging.debug(f"Processing row {idx} for user_id: {user_id}, asin: {asin}")
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
                logging.debug(f"Added aspect '{aspect}' with sentences: {' '.join(sentences)}")
    
    logging.info("Aspect tagging and expansion complete.")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Define the path to the data file
    file_path = "milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/cleaned_data_2018_2019.csv"

    # Load data
    logging.info(f"Loading data from {file_path}...")
    df_cleaned = pd.read_csv(file_path)
    logging.info("Data loaded successfully.")
    
    # Process data
    logging.info("Processing data for aspect tagging...")
    tagged_reviews_expanded = tag_and_expand_aspects(df_cleaned, aspects)
    
    # Display the resulting counts of aspects
    logging.info("Displaying aspect counts and df shape")
    print(tagged_reviews_expanded['aspect'].value_counts())
    print(tagged_reviews_expanded['aspect'].shape)
    tagged_reviews_expanded.to_csv('milestones/data-pipeline-requirements/03-data-preprocessing/staging/data/cleaned_data/aspect_based.csv')
    logging.info("Script execution completed.")
