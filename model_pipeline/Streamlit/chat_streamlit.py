import streamlit as st
from llm_fetch import process_text_query
import os

# Function to load values from text files in the specified folder
def load_filter_values(folder_path):
    """Load filter values from text files."""
    filters = {
        'categories': [],
        'subcategories': [],
        'years': [],
        'months': []
    }

    try:
        with open(os.path.join(folder_path, 'categories.txt'), 'r') as f:
            filters['categories'] = [line.strip() for line in f.readlines()]
        with open(os.path.join(folder_path, 'subcategories.txt'), 'r') as f:
            filters['subcategories'] = [line.strip() for line in f.readlines()]
        with open(os.path.join(folder_path, 'years.txt'), 'r') as f:
            filters['years'] = [line.strip() for line in f.readlines()]
        with open(os.path.join(folder_path, 'months.txt'), 'r') as f:
            filters['months'] = [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading filter values: {e}")
    
    return filters

# Streamlit app configuration
st.set_page_config(
    page_title="Amazon Reviews Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the folder path where the filter text files are stored
items_folder_path = 'model_pipeline/Streamlit/items'

# Load filter values from the text files
filters = load_filter_values(items_folder_path)

# Sidebar filters
st.sidebar.header("Filters")
category = st.sidebar.selectbox("Category", filters['categories'])
subcategory = st.sidebar.selectbox("Subcategory", filters['subcategories'])
year = st.sidebar.selectbox("Year", filters['years'])
month = st.sidebar.selectbox("Month", filters['months'])

# Main page header
st.title("Amazon Reviews Chatbot")
st.markdown(
    "This chatbot analyzes sentiment and provides insights based on product reviews. "
    "Use the filters in the sidebar for a more refined search."
)

# User input for chatbot query
user_input = st.text_input("Ask me a question about Amazon reviews:", "")

# Set top_k for results
top_k = st.sidebar.slider("Number of Results", min_value=1, max_value=20, value=10)

# Chatbot response
if st.button("Get Response"):
    if user_input.strip():
        with st.spinner("Processing your query..."):
            response = process_text_query(
                input_text=user_input,
                top_k=top_k,
                category=category,
                subcategory=subcategory,
                year=year,
                month=month,
            )
        st.subheader("Chatbot Response")
        st.write(response)
    else:
        st.warning("Please enter a question.")
