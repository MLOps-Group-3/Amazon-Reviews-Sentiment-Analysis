```markdown
# Amazon-Reviews-Sentiment-Analysis

### Team Members:
Valli Meenaa Vellaiyan <br>
Niresh Subramanian <br>
Venkata Subbarao Shirish Addaganti <br>
Harshit Sampgaon <br>
Prabhat Chanda <br>
Praneeth Korukonda <br>

---

## Introduction

In today's competitive e-commerce landscape, understanding customer feedback is critical for improving product offerings and enhancing the overall customer experience. This project focuses on leveraging sentiment analysis of Amazon reviews to gain deeper insights into customer sentiment. By categorizing reviews into positive, neutral, or negative, businesses like Amazon can extract actionable insights to inform product decisions, optimize customer service, and drive strategic initiatives aimed at improving customer satisfaction.

The goal of this project is to automate the end-to-end process of analyzing review data, from ingestion and pre-processing to model training, deployment, and monitoring. This automated solution not only increases operational efficiency but also helps Amazon to better understand how customers feel about their products and services. By analyzing sentiment trends over time, the model provides valuable insights into key pain points, emerging trends, and areas for improvement across different product categories.

Using advanced sentiment analysis, this project enables Amazon to identify critical feedback faster, enhance the customer experience, and make data-driven decisions that align with business goals such as increasing customer retention, boosting sales, and reducing return rates. This ultimately contributes to a more responsive and customer-focused business strategy, directly impacting business growth.

## Dataset Overview

### UCSD Amazon Reviews 2023
The UCSD Amazon Reviews 2023 dataset is a large, publicly available collection of customer reviews across multiple product categories on Amazon. It contains approximately **338 million reviews**, spanning diverse customer experiences and sentiments.

The dataset provides a foundation for sentiment analysis and is integral to our project's goal of classifying reviews into positive, neutral, or negative categories. The large volume and diversity of the dataset make it ideal for building scalable machine learning models.

### Citation
- **Dataset Source**: [UCSD Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html)
- **Citation**: Jérémie Rappaz, Julian McAuley, Karl Aberer. *Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption*, RecSys, 2021.

### Data Card
- **Format**: CSV/JSON
- **Size**: 338 million reviews
- **Data Types**: String, Numeric, List, Boolean, Dictionary, Timestamps
- **Key Features**:
  - **Review Text**: The main content of customer feedback
  - **Star Rating**: Ratings from 1 to 5 stars
  - **Product Category**: Product category for the reviewed item
  - **Review Timestamp**: Date and time of the review
  - **Product Metadata**: Additional product-related details
  - **Verified Purchase**: Indicator if the review is from a verified purchase
  - **Review Helpfulness**: Upvotes or downvotes received by the review (if available)

### Data Rights and Privacy
- The dataset is available for non-commercial use, and user identifiers like reviewer IDs are excluded to prevent privacy breaches. The project adheres to data minimization principles and complies with relevant privacy regulations.

## Folder Structure

```bash
.
├── .dvc                        # DVC configuration files for data versioning
├── .github/workflows           # GitHub Actions workflows for CI/CD
├── data_pipeline               # Main data pipeline directory
│   ├── archive                 # Deprecated files and early scripts
│   ├── config                  # Configuration files for pipeline settings
│   ├── dags                    # Airflow DAGs for pipeline stages
│   │   └── utils               # Utility scripts shared across DAGs
│   │       ├── data_collection # Data collection utilities
│   │       ├── data_preprocessing # Data preprocessing utilities
│   │       └── data_validation # Data validation utilities
│   ├── data                    # Directory for datasets
│   │   └── raw                 # Raw, unprocessed data
│   ├── logs                    # Airflow logs
│   │   └── scheduler           # Scheduler-specific logs
│   └── tests                   # Test suite for pipeline
│       ├── data_collection     # Tests for data collection
│       ├── data_preprocessing  # Tests for data preprocessing
│       └── data_validation     # Tests for data validation
├── data_pipeline_sampling_test # Alternative pipeline for sampling tests
│   ├── archive                 # Deprecated sampling-related files
│   ├── config                  # Sampling-specific configuration files
│   ├── dags                    # Sampling-specific DAGs
│   │   └── utils               # Shared sampling utilities
│   │       ├── data_collection
│   │       ├── data_preprocessing
│   │       └── data_validation
│   ├── data                    # Data directory for sampling
│   │   └── raw                 # Raw data for sampling
│   └── tests                   # Tests for sampling pipeline
│       ├── data_collection
│       ├── data_preprocessing
│       └── data_validation
├── milestones                  # Project milestone documents
│   ├── data-pipeline-requirements # Stage-wise pipeline requirements
│   │   ├── 01-data-collection  # Data collection stage details
│   │   │   ├── dags            # Stage-specific DAGs
│   │   │   ├── data            # Raw data files
│   │   │   │   └── raw
│   │   │   ├── logs            # Logs for the stage
│   │   │   │   └── scheduler
│   │   │   ├── plugins         # Plugins for the stage
│   │   │   ├── scripts         # Scripts for data collection
│   │   │   └── staging         # Staging data for this stage
│   │   ├── 02-data-validation
│   │   │   ├── Archive         # Archived validation files
│   │   │   └── staging         # Staging data for validation
│   │   │       ├── dags
│   │   │       │   └── utils
│   │   │       └── tests
│   │   └── 03-data-preprocessing
│   │       ├── archive         # Archived preprocessing files
│   │       │   ├── dags
│   │       │   │   └── utils
│   │       │   └── tests
│   │       └── staging         # Staging data for preprocessing
│   │           ├── dags
│   │           │   └── utils
│   │           ├── data
│   │           └── tests
│   └── scoping                 # Project scoping documents
├── model_deployment            # Model deployment workflows
│   ├── data_test               # Data for deployment tests
│   ├── endpoint_pytorchserve   # PyTorch Serve-specific deployment files
│   │   └── predictor           # Prediction-related scripts
│   │       └── utils
│   ├── pipeline_notebook       # Pipeline notebooks for deployment
│   │   ├── data_prep           # Data preparation scripts
│   │   │   └── utils
│   │   ├── predictor           # Prediction workflows
│   │   │   └── utils
│   │   └── trainer             # Model training scripts
│   │       └── utils
│   └── pipeline_notebook_v2    # Updated deployment workflows
│       └── src
│           └── utils
├── model_pipeline              # Machine learning pipelines
│   ├── rag                     # Retrieval-Augmented Generation pipeline
│   │   ├── config              # RAG configuration files
│   │   ├── dags                # DAGs for RAG
│   │   │   └── utils
│   │   └── data                # Data for RAG pipeline
│   ├── sentiment_model         # Sentiment analysis models
│   │   ├── assets              # Model assets
│   │   ├── dags                # Sentiment model DAGs
│   │   └── src                 # Source code for sentiment models
│   │       ├── mlruns          # MLflow tracking data
│   │       └── utils           # Utility scripts
│   ├── sentiment_model_on_kubeflow # Kubeflow integration for sentiment models
│   └── Streamlit               # Streamlit-based interactive dashboards
│       └── items               # Dashboard components
└── project_pipeline            # Integrated project pipeline
    ├── config                  # Configuration files
    ├── dags                    # Main DAGs for the project
    │   ├── data_utils          # Data utility scripts
    │   │   ├── data_collection
    │   │   ├── data_preprocessing
    │   │   └── data_validation
    │   ├── model_utils         # Model-related utilities
    │   │   └── src
    │   │       └── utils
    │   └── serve_utils         # Serving-related utilities
    ├── data                    # Project-wide data storage
    │   ├── labeled             # Labeled data for training and serving
    │   │   ├── serve
    │   │   └── train
    │   └── raw                 # Raw data
    └── tests                   # Project-wide test cases
        ├── data_collection
        ├── data_preprocessing
        └── data_validation
```

---

## Repository Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `pip` or `conda` for managing Python packages
- Docker (for containerized pipeline execution)
- DVC (Data Version Control)
- Apache Airflow
- Google Cloud SDK (for cloud storage and pipeline integration)

### Steps to Set Up

1. **Clone the repository:**
    ```bash
    git clone https://github.com/MLOps-Group-3/Amazon-Reviews-Sentiment-Analysis.git
    cd Amazon-Reviews-Sentiment-Analysis
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Airflow:**
    - Navigate to the `data_pipeline` directory and initialize Airflow:
        ```bash
        cd data_pipeline
        airflow db init
        airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com
        ```
    - Start the Airflow scheduler and webserver:
        ```bash
        airflow scheduler
        airflow webserver
        ```

4. **Run the pipeline:**
    - Trigger DAGs via the Airflow UI to execute stages of the pipeline (e.g., data acquisition, validation, preprocessing).

5. **Set up DVC:**
    - Pull the latest data files:
        ```bash
        dvc pull
        ```

6. **Configure GCP:**
    - Authenticate using the Google Cloud SDK:
        ```bash
        gcloud auth login
        ```
    - Set up your bucket for data storage and integration:
        ```bash
        gcloud storage buckets create gs://your-bucket-name
        ```

7. **Test the pipeline:**
    - Run the tests for data collection, validation, and preprocessing:
        ```bash
        pytest data_pipeline/tests
        ```

### Additional Information
- For detailed setup instructions for the `data_pipeline`, refer to [data_pipeline/README.md](data_pipeline/README.md).
- For setting up model pipelines and deployments, refer to the `model_pipeline` directory and its subfolders for specific configurations.

---

## Key Features

1. **End-to-End Pipeline Automation:**
    - Modular DAGs in Apache Airflow for ingestion, validation, preprocessing, and sentiment analysis.

2. **DVC Integration:**
    - Ensures version control for data and tracks changes across pipeline stages.

3. **Cloud Integration:**
    - Utilizes Google Cloud Platform (GCP) for scalable data storage and model deployment.

4. **Sentiment Analysis Models:**
    - Incorporates pre-trained models like BERT and RoBERTa for high-accuracy sentiment classification.

5. **RAG for Summarization:**
    - Implements Retrieval-Augmented Generation for aspect-wise summarization of reviews.

6. **Interactive Dashboards:**
    - Streamlit-based dashboards for visualization of insights and sentiment trends.

7. **CI/CD with GitHub Actions:**
    - Automated testing and deployment workflows for robust pipeline operations.

---

## Data Pipelines in Airflow

This repository contains Airflow DAGs for preprocessing, validating, and analyzing Amazon review data through a series of modular tasks. Each DAG corresponds to a distinct stage in the data pipeline, leveraging Python and Pandas for data transformation, sampling, validation, and sentiment analysis.

### Pipeline Stages Overview

1. **Data Acquisition:**
   - Extracts and ingests Amazon review data from the UCSD dataset.
   - Stores raw data for further processing.

2. **Data Sampling:**
   - Samples review data across specified categories to create balanced datasets.
   - Consolidates sampled data for downstream analysis.

3. **Data Validation:**
   - Ensures dataset quality and consistency by checking schema, null values, and integrity.
   - Logs invalid rows for further debugging and cleaning.

4. **Data Preprocessing:**
   - Cleans data by removing duplicates, handling missing values, and normalizing text.
   - Labels reviews with sentiment tags (positive, neutral, negative).

5. **Sentiment Analysis and Summarization:**
   - Applies pre-trained models like BERT for sentiment classification.
   - Implements RAG (Retrieval-Augmented Generation) for generating aspect-wise summaries.

### Key Tools and Frameworks
- **Apache Airflow:**
  - Orchestrates the pipeline with modular, reusable DAGs.
- **DVC:**
  - Tracks data changes and versions across pipeline stages.
- **MLflow:**
  - Manages model experiments and tracks metrics for sentiment analysis.

### DVC Integration
- The repository uses DVC for version control of datasets and intermediate outputs. Ensure you have the latest configuration pulled:
    ```bash
    dvc pull
    ```

---

## Model Pipeline

The `model_pipeline` directory contains workflows for training, evaluating, and deploying machine learning models for sentiment analysis and summarization.

### Key Features
1. **Sentiment Analysis Pipeline:**
   - Trains state-of-the-art models like BERT and RoBERTa.
   - Performs hyperparameter tuning and bias detection to optimize performance.

2. **RAG Pipeline:**
   - Uses a retrieval-based approach for generating aspect-wise summaries.
   - Handles large document contexts for better summarization accuracy.

3. **Deployment with PyTorch Serve:**
   - Deploys trained models as scalable REST APIs.
   - Includes endpoint configurations and predictor utilities.

4. **Interactive Dashboards:**
   - Built with Streamlit for exploring sentiment trends and insights dynamically.

### Deployment Setup
1. **Local Deployment:**
   - Use Docker to containerize and run the model locally.
   - Navigate to the `model_deployment` folder and follow the instructions in the README file.

2. **Cloud Deployment:**
   - Use Google Cloud's Vertex AI or PyTorch Serve for scalable deployment.
   - Ensure your GCP project and credentials are properly configured.

### Tools and Integrations
- **MLflow**: Tracks model metrics, artifacts, and experiments.
- **Vertex AI**: Deploys trained models and monitors their performance in the cloud.
- **Apache Airflow**: Manages the orchestration of modular pipelines.

---
