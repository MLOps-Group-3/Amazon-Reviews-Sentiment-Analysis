import streamlit as st
import json
from llm_fetch import process_text_query
import streamlit.components.v1 as components

# Function to load hierarchical metadata from JSON file
def load_hierarchical_metadata(file_path):
    """Load hierarchical metadata from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return {}

# Function to get available options dynamically based on selected filters
def get_available_options(metadata, category=None, year=None):
    """Get dynamic options based on selected hierarchy."""
    if category is None:
        return list(metadata.keys())
    if year is None:
        return list(metadata.get(category, {}).keys())
    return metadata.get(category, {}).get(year, [])

# Streamlit app configuration
st.set_page_config(
    page_title="Amazon Reviews Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path to the hierarchical metadata file
metadata_file_path = '/Users/praneethkorukonda/Documents/Amazon-Reviews-Sentiment-Analysis/model_pipeline/Streamlit/items/hierarchical_metadata.json'

# Load metadata from the JSON file
metadata = load_hierarchical_metadata(metadata_file_path)

# Sidebar filters
st.sidebar.header("Filters")

# Navigation
page = st.sidebar.radio("Select a Page", ("Chatbot", "Sentiment Analysis"))

# Chatbot Page
if page == "Chatbot":
    st.title("Amazon Reviews Chatbot")
    st.markdown(
        "This chatbot analyzes sentiment and provides insights based on product reviews. "
        "Use the filters in the sidebar for a more refined search."
    )
    
    # Dynamic category selection
    selected_category = st.sidebar.selectbox("Category", get_available_options(metadata))
    if selected_category:
        # Dynamic year selection based on category
        selected_year = st.sidebar.selectbox(
            "Year",
            get_available_options(metadata, selected_category)
        )
        if selected_year:
            # Dynamic month selection based on category and year
            selected_month = st.sidebar.selectbox(
                "Month",
                get_available_options(metadata, selected_category, selected_year)
            )
        else:
            selected_month = None
    else:
        selected_year = None
        selected_month = None

    # User input for chatbot query
    user_input = st.text_input("Ask me a question about Amazon reviews:", "")

    # Chatbot response
    if st.button("Get Response"):
        if user_input.strip():
            with st.spinner("Processing your query..."):
                response = process_text_query(
                    input_text=f"Find results for the query '{user_input}' in the '{selected_category}' category for the year {selected_year} and the month of {selected_month}",
                    category=selected_category,
                    year=selected_year,
                    month=selected_month,
                )
            st.subheader("Chatbot Response")
            st.write(response)
        else:
            st.warning("Please enter a question.")

# Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.title("Product Sentiment Analysis")
    
    # Embed Tableau visualization using components
    tableau_url = "https://public.tableau.com/views/MARRYLANDYEARLYROADCRASHREPORT-GROUP26-PROJECT3/MarylandYearlyRoadCrashReport"
    components.html(f"""
    <iframe src="{tableau_url}" width="100%" height="800px" frameborder="0"></iframe>
    """, height=800)
