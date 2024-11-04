# Amazon-Reviews-Sentiment-Analysis

### Team Members:

Valli Meenaa Vellaiyan <br>
Niresh Subramanian <br>
Venkata Subbarao Shirish Addaganti <br>
Harshit Sampgaon <br>
Prabhat Chanda <br>
Praneeth Korukonda <br>

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
├── data_pipeline                  # Main directory for all data pipeline assets and configurations
│   ├── archive                    # Contains deprecated files and early versions of scripts
│   │   ├── docker-compose-collection.yaml.txt  
│   │   ├── docker-compose python.txt
│   │   └── sampling_old.py                  
│   ├── config                     # Directory for configuration files
│   │   └── config.ini             # Configuration file for pipeline settings and parameters
│   ├── dags                       # Directory containing all Airflow DAGs
│   │   ├── data_acquisition_dag.py   # DAG for data acquisition and ingestion
│   │   ├── data_preprocessing_dag.py # DAG for data cleaning, labeling, and sentiment analysis
│   │   ├── data_validation_dag.py    # DAG for data validation to ensure quality and consistency
│   │   ├── __init__.py               # Initialization for the DAGs package
│   │   ├── sampling_dag.py           # DAG for sampling Amazon review data by category
│   │   └── utils                     # Utility scripts used across DAGs for modular functionality
│   ├── data                       # Data directory containing datasets across different pipeline stages
│   │   ├── cleaned                # Cleaned data generated in preprocessing
│   │   ├── cleaned.dvc            # DVC tracking file for cleaned data directory
│   │   ├── labeled                # Labeled data with sentiment tags
│   │   ├── labeled.dvc            # DVC tracking file for labeled data directory
│   │   ├── raw                    # Raw, unprocessed data files from data acquisition
│   │   ├── sampled                # Sampled data across categories
│   │   ├── sampled.dvc            # DVC tracking file for sampled data directory
│   │   ├── validation             # Validated data directory, storing outputs from data validation
│   │   └── validation.dvc         # DVC tracking file for validation data directory
│   ├── docker-compose.yaml        # Docker Compose file to set up the environment for the pipeline
│   ├── Dockerfile                 # Dockerfile for building the application environment
│   ├── __init__.py                # Initialization file for data_pipeline package
│   ├── logs                       # Log directory for Airflow task and DAG execution logs
│   │   ├── dag_id=...             # Log subdirectories for each DAG by DAG ID
│   ├── plugins                    # Directory for custom Airflow plugins if needed
│   ├── README.md                  # Detailed README for data pipeline configuration
│   ├── requirements.txt           # List of Python dependencies for the pipeline
│   └── tests                      # Test suite for different stages of the data pipeline
│       ├── data_collection        # Tests for data collection functionality
│       ├── data_preprocessing     # Tests for data preprocessing tasks
│       ├── data_validation        # Tests for validation steps
│       └── __init__.py            # Initialization file for tests
├── LICENSE                        # License for the repository
├── milestones                     # Archived milestone files from earlier project stages
│   ├── data-pipeline-requirements # Requirements and specifications for each data pipeline stage
│   ├── scoping                    # Scoping documents for project planning
├── model_pipeline                 # Placeholder for model-specific pipeline assets
└── README.md                      # Main README for repository overview and usage instructions
```

## Repository Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `pip` or `conda` for managing Python packages

### Steps

1. #### Clone the repository:
    ```bash
    git clone https://github.com/MLOps-Group-3/Amazon-Reviews-Sentiment-Analysis.git
    cd Amazon-Reviews-Sentiment-Analysis
    ```

<!-- For setting up the `data_pipeline`, please refer to [data_pipeline/README.md](data_pipeline/README.md). -->
2. #### Data Pipeline Setup

    To run the services in Docker, navigate to the `data_pipeline` directory and follow the instructions to build the Docker image and start Docker Compose.

    For detailed setup instructions, including installing dependencies and configuring the environment for the `data_pipeline`, refer to [data_pipeline/README.md](data_pipeline/README.md).








# Data Pipelines in Airflow

This repository contains Airflow DAGs for preprocessing, validating, and analyzing Amazon review data through a series of modular tasks. Each DAG corresponds to a distinct stage in the data pipeline, leveraging Python and Pandas for data transformation, sampling, validation, and sentiment analysis.

## Pipeline Stages Overview

1. **Data Acquisition**: Extracts and ingests Amazon review data.
2. **Data Sampling**: Samples review data across specified categories, consolidating it for further processing.
3. **Data Validation**: Validates dataset quality and structure, ensuring integrity for downstream analysis.
4. **Data Preprocessing**: Cleans, labels, and tags aspects within review data for sentiment analysis.

Each DAG stage runs independently, enabling focused transformations while maintaining a sequential flow between acquisition, sampling, validation, and preprocessing.

### DVC Integration

This project uses DVC for version control of data files, with storage configured via Google Cloud. Ensure you have the required credentials and configuration in place to pull the latest data files.

For a detailed breakdown of each DAG and setup instructions, refer to the [data_pipeline/README.md](data_pipeline/README.md).
