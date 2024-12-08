import streamlit as st
import json
from llm_fetch import process_text_query
from dotenv import load_dotenv
import os

load_dotenv()

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
    page_title="Amazon Reviews Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path to the hierarchical metadata file
metadata_file_path = "/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/model_pipeline/Streamlit/items/hierarchical_metadata.json"

# Load metadata from the JSON file
metadata = load_hierarchical_metadata(metadata_file_path)

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Page", ("Introduction", "Summary Generator","Model Test"))

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #333;
    }
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .intro-text {
        font-size: 16px;
        color: #555;
        line-height: 1.7;
    }
    .highlight {
        font-weight: bold;
        color: #1abc9c;
    }
    .sub-header {
        font-size: 22px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #95a5a6;
        margin-top: 50px;
    }
    .btn-primary {
        background-color: #1abc9c;
        color: white;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Introduction Page
if page == "Introduction":
    # Title and Header Section
    st.markdown('<div class="main-header">📊 Welcome to the Amazon Reviews Dashboard</div>', unsafe_allow_html=True)

    # GitHub Repository Link
    st.markdown("""
        <div class="intro-text">
            For more details, please visit our 
            <a href="https://github.com/MLOps-Group-3/Amazon-Reviews-Sentiment-Analysis/tree/main" 
            target="_blank" class="highlight">GitHub Repository</a>, which contains the full end-to-end sentiment analysis pipeline.
        </div>
    """, unsafe_allow_html=True)

    # Dashboard Description
    st.markdown("""
    <div class="intro-text">
        Welcome to the <span class="highlight">Amazon Reviews Dashboard</span>, a comprehensive platform designed to extract actionable insights from customer feedback. This tool empowers stakeholders to:
    </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
    <ul class="intro-text">
        <li>📈 Analyze trends and sentiment insights effectively.</li>
        <li>🛠️ Refine data exploration using dynamic filters like <span class="highlight">Category</span>, <span class="highlight">Year</span>, and <span class="highlight">Month</span>.</li>
        <li>💡 Generate data-driven summaries focusing on performance, quality, and customer satisfaction metrics.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Usage Instructions
    st.markdown("""
    <div class="intro-text">
        Use the sidebar to navigate to the <span class="highlight">Summary Generator</span> tab for data insights. Apply the filters provided and click "Proceed" to generate insights tailored to your requirements.
    </div>
    """, unsafe_allow_html=True)

    # Footer Section
    st.markdown("""
    <div class="footer">
        Designed for data-driven decision-making by MLOps-Group-3.
    </div>
    """, unsafe_allow_html=True)

# Summary Generator Page
elif page == "Summary Generator":
    st.markdown('<div class="main-header">Amazon Reviews Summary Generator</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='intro-text'>Select the filters below to generate a summary of Amazon reviews based on the chosen category, year, and month.</div>",
        unsafe_allow_html=True,
    )

    # Dynamic category selection
    selected_category = st.selectbox("Category", get_available_options(metadata))
    selected_year = None
    selected_month = None

    if selected_category:
        # Dynamic year selection based on category
        selected_year = st.selectbox(
            "Year",
            get_available_options(metadata, selected_category)
        )
        if selected_year:
            # Dynamic month selection based on category and year
            selected_month = st.selectbox(
                "Month",
                get_available_options(metadata, selected_category, selected_year)
            )

    # Process Button
    if st.button("Proceed", key="proceed", help="Click to generate the summary"):
        if selected_category and selected_year and selected_month:
            with st.spinner("Generating summary..."):
                response = process_text_query(
                    input_text=f"Generate a summary for the '{selected_category}' category for the year {selected_year} and the month of {selected_month}.",
                    category=selected_category,
                    year=selected_year,
                    month=selected_month,
                )
            st.markdown('<div class="sub-header">Summary</div>', unsafe_allow_html=True)
            st.success(response)
        else:
            st.warning("Please select all filters (Category, Year, and Month) to proceed.")

# Model Test Page
elif page == "Model Test":
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    service_key_path = os.getenv("GCS_SERVICE_ACCOUNT_KEY_LOCAL")
    credentials = service_account.Credentials.from_service_account_file(service_key_path)

    st.markdown('<div class="main-header">Test Your Model</div>', unsafe_allow_html=True)
    st.markdown("<div class='intro-text'>Provide custom inputs below to test the deployed model and get predictions.</div>", unsafe_allow_html=True)

    # Input fields for model testing
    text_input = st.text_area("Text Input", "The product was excellent and exceeded my expectations.", help="Enter the review text.")
    price_input = st.number_input("Price", value=49.99, help="Enter the price of the product.")
    price_missing_input = st.selectbox("Is Price Missing?", [False, True], help="Select whether the price information is missing.")
    helpful_vote_input = st.number_input("Helpful Votes", value=15, help="Enter the number of helpful votes.")
    verified_purchase_input = st.selectbox("Verified Purchase?", [True, False], help="Select whether the purchase is verified.")

    # Initialize AI Platform
    PROJECT_ID = os.getenv("PROJECT_ID","amazonreviewssentimentanalysis")  # Replace with your project ID or set it in .env
    REGION = os.getenv("GCP_REGION", "us-central1")  # Replace with your region or set in .env
    APP_NAME = "review_sentiment_bert_model"  # Replace with your app name

    aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)

    # Construct endpoint display name
    endpoint_display_name = f"{APP_NAME}-endpoint"
    filter = f'display_name="{endpoint_display_name}"'

    # Retrieve the endpoint
    endpoint_info = None
    for endpoint_info in aiplatform.Endpoint.list(filter=filter):
        print(
            f"Endpoint display name = {endpoint_info.display_name}, resource ID = {endpoint_info.resource_name}"
        )

    if endpoint_info:
        endpoint = aiplatform.Endpoint(endpoint_info.resource_name)
        print(f"Endpoint ready for use: {endpoint_info.resource_name}")
    else:
        # Create endpoint if it doesn't exist
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
        print(f"Created new endpoint: {endpoint.display_name}")

    # Check if the model is already deployed
    is_model_deployed = False
    deployed_models = endpoint.list_models()
    for deployed_model in deployed_models:
        
        if deployed_model.display_name == APP_NAME+"-v1":
            is_model_deployed = True
            print(f"Model '{APP_NAME}' is already deployed to this endpoint.")
            break

    # Deploy the model if not deployed
    if not is_model_deployed:
        VERSION = 1
        model_display_name = f"{APP_NAME}-v{VERSION}"

        st.info(f"Deploying model '{APP_NAME}' to endpoint...")
        model = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')[0]
        
        traffic_percentage = 100
        machine_type = "n1-standard-4"
        deployed_model_display_name = model_display_name
        min_replica_count = 1
        max_replica_count = 3
        sync = True

        # Deploy the model
        operation = model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=deployed_model_display_name,
            machine_type=machine_type,
            traffic_percentage=traffic_percentage,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            sync=sync,
        )
        st.success(f"Model '{APP_NAME}' successfully deployed to endpoint '{endpoint_display_name}'!")
    else:
        st.success(f"Model '{APP_NAME}' is already deployed to endpoint '{endpoint_display_name}'.")

    # Predict button
    if st.button("Get Prediction"):
        with st.spinner("Fetching prediction..."):
            try:
                # Construct the input instance
                instance = {
                    "text": text_input,
                    "price": price_input,
                    "price_missing": price_missing_input,
                    "helpful_vote": helpful_vote_input,
                    "verified_purchase": verified_purchase_input,
                }

                # Send the prediction request
                response = endpoint.predict(instances=[instance])

                # Display the prediction results
                st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
                sentiment = response.predictions[0][0]
                confidence = response.predictions[0][1]
                print(sentiment,confidence)
                # Determine color based on sentiment
                if sentiment == "NEGATIVE":
                    sentiment_color = "red"
                elif sentiment == "NEUTRAL":
                    sentiment_color = "orange"
                else:
                    sentiment_color = "green"

                # Display sentiment with color
                st.markdown(
                    f"""
                    <div style="font-size:20px; font-weight:bold; color:{sentiment_color};">
                        Sentiment: {sentiment}
                    </div>
                    <div style="font-size:16px; color:gray;">
                        Confidence: {confidence}
                    </div>
                    """,
                    unsafe_allow_html=True, 
                    )   
            except Exception as e:
                st.error(f"An error occurred: {e}")
