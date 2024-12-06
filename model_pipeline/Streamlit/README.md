# Streamlit Interface for Amazon Reviews Sentiment Analysis

This directory contains the Streamlit-based stakeholder interface for summarizing product category sentiments. The interface integrates RAG (Retrieval-Augmented Generation) pipelines and provides category-based sentiment summaries using Pinecone for vector storage and OpenAI's models for LLM-based summarization where the data are stored in GCS.

## Directory Structure

```
.
├── __pycache__/
│   ├── llm_fetch.cpython-310.pyc
│   ├── llm_fetch.cpython-313.pyc
│   └── llm_fetch.cpython-39.pyc
├── amazonreviewssentimentanalysis-8dfde6e21c1d.json  # GCP authentication credentials
├── chat_streamlit.py                                # Streamlit application for stakeholder interaction
├── item_store_extractor.py                          # Script to extract and manage item metadata
├── items/                                           # Metadata for categories, subcategories, and temporal data
│   ├── categories.txt
│   ├── hierarchical_metadata.json
│   ├── months.txt
│   ├── subcategories.txt
│   └── years.txt
├── llm_fetch.py                                     # Utility for fetching summaries using LLMs and Pinecone
├── readme.md                                        # Documentation for this interface
└── test.py                                          # Script for testing the interface components
```

## Overview

This interface is designed as part of the **RAG DAG pipeline** to provide stakeholders with category-specific sentiment analysis and insights into Amazon product reviews. The application fetches data from a document store, processes it through LLMs, and displays results interactively.

### Features

- **Category-Based Sentiment Summarization**:
  - Stakeholders can select product categories or subcategories to view summaries of sentiments.
  - Data is aggregated and filtered by category, subcategory, and time (year/month).
  
- **Dynamic Data Management**:
  - Metadata files like  `hierarchical_metadata.json` allow for flexible updates to the product category hierarchy.
  
- **Integration with Pinecone and OpenAI**:
  - Pinecone is used for efficient vector search and storage.
  - OpenAI models power the LLM-based summarization for generating insights.

- **Streamlit Dashboard**:
  - Provides an intuitive and interactive interface for stakeholders to explore results.

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-folder>/model_pipeline/Streamlit
   ```

2. **Install Dependencies**:

   Create a virtual environment and install the required dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add Credentials**:

   Place the GCP authentication file (`amazonreviewssentimentanalysis-8dfde6e21c1d.json`) in the directory.

4. **Run the Application**:

   Launch the Streamlit app using:

   ```bash
   streamlit run chat_streamlit.py
   ```

5. **Test the Setup** (Optional):

   Run `test.py` to ensure all components work as expected:

   ```bash
   python test.py
   ```

## Files Description

- **`chat_streamlit.py`**: Main application file for the Streamlit interface.
- **`item_store_extractor.py`**: Extracts item metadata, including categories and time-based information, for use in the interface.
- **`llm_fetch.py`**: Handles interaction with OpenAI's LLM and Pinecone to fetch sentiment summaries.
- **`items/`**:
  - `categories.txt`, `subcategories.txt`, `months.txt`, `years.txt`: Text files containing categorical and temporal metadata.
  - `hierarchical_metadata.json`: JSON file mapping categories to subcategories for dynamic filtering.
- **`test.py`**: Script to validate the functionality of different components.
- **`amazonreviewssentimentanalysis-8dfde6e21c1d.json`**: GCP authentication credentials for accessing cloud services.

