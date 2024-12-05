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
page = st.sidebar.radio("Select a Page", ("Introduction", "Summary Generator"))


# Introduction Page
if page == "Introduction":
    # Title and Header Section
    st.title("📊 Welcome to the Amazon Reviews Dashboard")
    st.markdown("""
        <style>
        .intro-header {
            font-size: 22px;
            font-weight: bold;
            color: #4CAF50;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .intro-text {
            font-size: 18px;
            color: #333;
            line-height: 1.6;
        }
        .highlight {
            font-weight: bold;
            color: #2E8B57;
        }
        </style>
        """, unsafe_allow_html=True)

    # Image Header
    st.image(
        "https://via.placeholder.com/800x200",
        caption="Amazon Reviews Insights Dashboard",
        use_column_width=True
    )

    # Dashboard Description
    st.markdown("""
    <div class="intro-text">
        Welcome to the <span class="highlight">Amazon Reviews Dashboard</span>, a comprehensive platform designed to provide actionable insights from customer feedback. This dashboard empowers you to:
    </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
    <ul class="intro-text">
        <li>📈 Explore review trends and key sentiment insights.</li>
        <li>🛠️ Refine your analysis using dynamic filters like <span class="highlight">Category</span>, <span class="highlight">Year</span>, and <span class="highlight">Month</span>.</li>
        <li>💡 Generate summaries that highlight performance, quality, and customer satisfaction.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Usage Instructions
    st.markdown("""
    <div class="intro-text">
        Navigate to the <span class="highlight">Summary Generator</span> tab using the sidebar to explore data insights. Start by selecting filters and click "Proceed" to view the summaries.
    </div>
    """, unsafe_allow_html=True)

    # Call-to-Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Learn More"):
            st.markdown("""
            <div class="intro-text">
                Explore detailed documentation and understand how this dashboard was built.
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if st.button("🚀 Get Started"):
            st.markdown("""
            <div class="intro-text">
                Jump straight into exploring reviews and generating insights!
            </div>
            """, unsafe_allow_html=True)

    # Footer Section
    st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 14px; color: #666;">
        Built with ❤️ using Streamlit | Designed for data-driven decision-making.
    </div>
    """, unsafe_allow_html=True)

# Summary Generator Page
elif page == "Summary Generator":
    st.title("Amazon Reviews Summary Generator")
    st.markdown(
        "Select the filters below to generate a summary of Amazon reviews based on the chosen category, year, and month."
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
    if st.button("Proceed"):
        if selected_category and selected_year and selected_month:
            with st.spinner("Generating summary..."):
                response = process_text_query(
                    input_text=f"Generate a summary for the '{selected_category}' category for the year {selected_year} and the month of {selected_month}.",
                    category=selected_category,
                    year=selected_year,
                    month=selected_month,
                )
            st.subheader("Summary")
            st.success(response)
        else:
            st.warning("Please select all filters (Category, Year, and Month) to proceed.")