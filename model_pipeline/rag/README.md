# Retrieval-Augmented Generation (RAG) Pipeline

## Overview

This repository contains a Retrieval-Augmented Generation (RAG) pipeline implemented using Apache Airflow for data orchestration, combined with OpenAI's GPT model for text summarization. The project is structured to handle large-scale document processing efficiently, making it suitable for production-level use cases.

---

## Features

1. **Data Preprocessing**:
   - Load, clean, and aggregate review data.
   - Prepare documents for RAG-based summarization.
   - Store processed data in structured formats.

2. **Document Processing**:
   - Utilize OpenAI's GPT to generate summaries.
   - Ensure data validation and clean the output.
   - Save refined documents for downstream use.

3. **Airflow Orchestration**:
   - Modular tasks grouped using Airflow's `TaskGroup`.
   - Extensive logging and error handling for reliability.
   - Flexible, production-standard DAG configuration.

---

## Directory Structure

```plaintext
.
├── Dockerfile
├── config
│   └── config.ini
├── dags
│   ├── rag_data_preprocessing_dag.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── document_processor.py
│       ├── embedings.py
│       ├── extracting_unique_aspects.py
│       ├── load_utils.py
│       ├── main.py
│       ├── prompt.txt
│       ├── review_data_processing.py
│       ├── summarization.py
│       └── summary_generator_main.py
├── docker-compose.yaml
└── requirements.txt                           # Environment variables
```

---

## Installation

### Prerequisites
- **Docker** and **Docker Compose** installed.
- **OpenAI API Key** for GPT integration.
- **Python 3.10+** installed for local development and debugging.

### Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```plaintext
   OPENAI_API_KEY=<your_openai_api_key>
   ```

3. **Build and Start Containers**:
   ```bash
   docker-compose up --build
   ```

4. **Access Airflow UI**:
   - Open your browser and navigate to [http://localhost:8080](http://localhost:8080).
   - Default login credentials: `airflow / airflow`.

---

## Usage

1. **Trigger the DAG**:
   - Enable and trigger the `rag_data_preprocessing_dag` from the Airflow UI.

2. **Pipeline Workflow**:
   - **Data Preprocessing**:
     - Load raw data, clean, and aggregate for document preparation.
   - **Document Processing**:
     - Summarize documents using OpenAI's GPT.
     - Validate and refine the JSON output.

3. **Outputs**:
   - Processed documents are saved to `REFINED_PROCESSED_DATA_PATH` for further analysis or use.

---

## Key Components

- **DAG File**:
  - `rag_data_preprocessing_dag.py`: Defines the workflow using Airflow to manage each stage of the RAG pipeline.

- **Utility Scripts**:
  - `review_data_processing.py`: Contains functions to load, clean, and aggregate data.
  - `document_processor.py`: Handles document summarization using GPT and cleans the final output.
  - `config.py`: Configuration settings including file paths.

- **Prompt Template**:
  - `prompt.txt`: Template used for generating prompts for OpenAI GPT to ensure consistency across documents.

---

## Dependencies

List of dependencies is included in `requirements.txt`:
```plaintext
apache-airflow==2.10.2
pandas
openai
python-dotenv
transformers
scikit-learn
```

To install them, run:
```bash
pip install -r requirements.txt
```

---

## Environment Variables

All necessary environment variables should be defined in a `.env` file:
```plaintext
OPENAI_API_KEY=<your_openai_api_key>
```

Ensure the `.env` file is present in the root directory for proper configuration.
