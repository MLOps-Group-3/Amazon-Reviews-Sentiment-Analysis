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


## Folder Structure

```bash
├── README.md                   # Overview of the project, installation instructions, and usage guidelines
├── LICENSE                     # License outlining terms and conditions for using the repository
├── .github/                    # GitHub-related files (e.g., workflows, templates, GitHub actions)
│   └── workflows/              # CI/CD workflows (e.g., GitHub actions)
├── data/                       # Raw and unprocessed data files
├── models/                     # Saved machine learning models and related metadata
├── notebooks/                  # Jupyter/Colab notebooks for exploration, prototyping, and modeling
├── scripts/                    # Deployment and monitoring scripts
├── src/                        # Source code for data processing, modeling, evaluation, and DAGs for Airflow
├── tests/                      # Unit tests for scripts and methods
└── milestones/                 # Files logging milestones and project progress
    └── scoping/                # Scoping documents and reports
```
## Initial Installation Requirements (Updated with progress)

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `pip` or `conda` for managing Python packages
- Docker for containerized environments

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/MLOps-Group-3/Amazon-Reviews-Sentiment-Analysis.git

    cd Amazon-Reviews-Sentiment-Analysis
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If using Docker:
    ```bash
    docker build -t Amazon-Reviews-Sentiment-Analysis .
    docker run -it -v $(pwd):/app Amazon-Reviews-Sentiment-Analysis
    ```
## Infrastructure Setup with Terraform on GCP

This project includes a Terraform configuration to create the necessary Google Cloud infrastructure.

### Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) (v1.0+)
- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install) installed and configured
- A Google Cloud account with billing enabled

### Steps

1. **Clone the repository** (if not done already):
    ```bash
    git clone https://github.com/Amazon-Reviews-Sentiment-Analysis/Amazon-Reviews-Sentiment-Analysis.git
    cd Amazon-Reviews-Sentiment-Analysis
    ```

2. **Navigate to the `terraform` directory**:
    ```bash
    cd infrastructure/terraform
    ```

3. **Authenticate with GCP**:
    ```bash
    gcloud auth application-default login
    gcloud config set project [PROJECT_ID]
    ```

4. **Initialize Terraform**:
    This step initializes the configuration and downloads necessary GCP provider plugins:
    ```bash
    terraform init
    ```

5. **Create and apply the infrastructure plan**:
    - To see the infrastructure that will be created:
      ```bash
      terraform plan
      ```
    - To create resources on GCP:
      ```bash
      terraform apply
      ```

6. **Destroy the infrastructure (optional)**:
    When done, to tear down the infrastructure:
    ```bash
    terraform destroy
    ```

### Usage

To run the scripts for training the model:
```bash
python src/train_model.py
