# Model Pipeline Overview

This repository contains multiple pipelines designed for handling sentiment analysis and document summarization at scale. It combines Sentiment Analysis, Vertex AI, and Retrieval-Augmented Generation (RAG) for cloud-based deployment and efficient document processing workflows.

---

## Pipelines

### 1. Sentiment Analysis Pipeline

This pipeline focuses on training, evaluating, and monitoring a sentiment analysis model. It includes hyperparameter tuning, model evaluation, bias detection, and integrates with MLflow for experiment tracking.

**Features**:
- **Data Preparation**: Cleans and prepares data for model training.
- **Model Training & Evaluation**: Utilizes BERT or RoBERTa for sentiment analysis and tracks performance metrics.
- **Bias Detection**: Identifies and mitigates biases in the model.
- **Experiment Tracking**: Integrates with MLflow for tracking model metrics, artifacts, and experiments.
- **Flexible Execution**: Supports retriggers for underperforming models based on metrics like F1 score.

**Key Components**:
- **DAG File**: `mlflow_tracking_dag.py`
- **Utility Scripts**: `prepare_data.py`, `train_save.py`, `bias_detect.py`

**Setup**:
- Dockerized environment for easy setup and testing.
- Run the pipeline using Airflow for modular execution and monitoring.

### 2. Vertex AI Pipeline

This pipeline leverages Google Cloud’s Vertex AI for scalable sentiment analysis. It automates model training, evaluation, and deployment, ensuring seamless integration with cloud resources.

**Features**:
- **Google Cloud Integration**: Utilizes Vertex AI for model training and deployment.
- **Dataset Registration**: Registers datasets in Vertex AI for streamlined processing.
- **Pipeline Execution**: Automates the entire pipeline from model training to deployment.
- **Cloud Resource Management**: Ensures optimal performance by utilizing GCP resources.

**Key Components**:
- **Pipeline Script**: `vertex_ai_pipeline.py`
- **Setup Script**: `vertex_ai_setup.py`

**Setup**:
- Configure GCP credentials and enable necessary APIs.
- Set up a GCS bucket for storing pipeline artifacts.
- Run the pipeline on Google Cloud for scalable execution.

### 3. Retrieval-Augmented Generation (RAG) Pipeline

The RAG pipeline integrates Apache Airflow for orchestration and OpenAI's GPT model for document summarization. This pipeline is ideal for large-scale document processing, ensuring efficient data handling and flexible workflow management.

**Features**:
- **Data Preprocessing**: Aggregates and cleans review data for summarization.
- **Document Summarization**: Uses OpenAI's GPT to generate summaries based on aggregated data.
- **Airflow Orchestration**: Modular tasks with error handling and flexible DAG configuration.

**Key Components**:
- **DAG File**: `rag_data_preprocessing_dag.py`
- **Utility Scripts**: `review_data_processing.py`, `document_processor.py`, `summarization.py`

**Setup**:
- Dockerized environment for local testing and development.
- Trigger and monitor the DAG from the Airflow UI.



## Installation & Setup

### Prerequisites:

- **Docker** and **Docker Compose** for containerized execution.
- **Google Cloud SDK** for setting up Vertex AI.
- **Python 3.9+** and **Conda** for local development.
- **OpenAI API Key** for GPT integration.

### Setup Instructions:

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd model_pipeline
   ```

2. **Set Up Environment**:
   - Create `.env` for environment variables (e.g., OpenAI API key, GCP credentials).
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run Docker Containers**:
   ```bash
   docker-compose up --build
   ```

## Experiment Tracking & Monitoring

- **MLflow**: Used for tracking experiments, metrics, and models for the Sentiment Analysis pipeline.
- **Airflow UI**: Used for monitoring and triggering the RAG pipeline.
- **Vertex AI Console**: Used to monitor the execution and performance of the Vertex AI pipeline.
