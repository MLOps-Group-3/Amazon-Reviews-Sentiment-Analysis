# Vertex AI Pipeline Setup and Execution

This guide provides step-by-step instructions to set up and execute the `vertex_ai_pipeline.py` script for running a Vertex AI pipeline.

## Prerequisites

### Tools and Services Required:
1. **Python 3.9**: Ensure Python 3.9 is installed on your system.
2. **Conda**: For virtual environment management.
3. **Google Cloud SDK (gcloud)**: To authenticate and interact with Google Cloud Platform.
4. **Required Permissions**:
   - Access to a GCP project with Vertex AI APIs enabled.
   - IAM roles: `Vertex AI Admin`, `Storage Admin`.

## Steps to Set Up and Run the Pipeline

### 1. Setting Up Google Cloud

1. **Install Google Cloud SDK**  
   Follow [these instructions](https://cloud.google.com/sdk/docs/install) to install the Google Cloud SDK on your system.

2. **Authenticate with Google Cloud**  
   Log in to your Google Cloud account:
   ```bash
   gcloud auth login

3. **Set the GCP Project**  
   Replace `<PROJECT_ID>` with your project ID:
   ```bash
   gcloud config set project <PROJECT_ID>

4. **Enable Required APIs**  
   Enable the Vertex AI and Cloud Storage APIs:
   ```bash
   gcloud services enable aiplatform.googleapis.com storage.googleapis.com

### 2. Creating the Virtual Environment

1. **Create a Conda Environment**  
   Create a Conda virtual environment with Python 3.9:
   ```bash
   conda create --name vertex-env python=3.9 -y

2. **Activate the Environment**  
   Activate the Conda environment:
   ```bash
   conda activate vertex-env

3. **Install Dependencies**
   pip install -r requirements.txt

### 3. Configure GCP for the Pipeline

1. **Set Up Authentication for the Script**  
   Ensure your `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to your service account key JSON file:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-key.json"

2. **Create a GCS Bucket**  
   Create a bucket to store pipeline artifacts:
   ```bash
   gcloud storage buckets create gs://<YOUR_BUCKET_NAME> --region=<REGION>

3. **Update `vertex_ai_pipeline.py`**  
   Modify the script to include your GCP project ID, bucket name, and region:
   ```python
   PROJECT_ID = "your-gcp-project-id"
   BUCKET_NAME = "your-bucket-name"
   REGION = "your-region"

### 4. Running the Pipeline

1. **Navigate to the Project Directory**  
   Change to the directory where the pipeline script is located:
   ```bash
   cd ~/Amazon-Reviews-Sentiment-Analysis/data_pipeline/vertex-ai-updated

2. **Run the Dataset Registration Script**  
   Run the `vertex_ai_setup.py` script to register your dataset in Vertex AI:
   ```bash
   python vertex_ai_setup.py

This script will:
   - Initialize the Vertex AI environment with your project and region.
   - Create a Tabular Dataset from the CSV file stored in GCS.
   - Output the dataset ID and resource name, which can be used in the pipeline script.

3. **Run the Pipeline Script**  
   Execute the script:
   ```bash
   python vertex_ai_pipeline.py

4. **Monitor the Pipeline Execution**  
   Use the Vertex AI console to monitor the pipeline:
   - Go to the [Vertex AI Pipelines](https://console.cloud.google.com/vertex-ai/pipelines) page in the GCP Console.

## Additional Notes

- **Troubleshooting**:
  - If modules are missing, install them using `pip install <module-name>`.
  - Ensure the Python version and package dependencies are compatible.
- **Customizations**:
  - Update the `requirements.txt` file to include any additional dependencies.
  - Modify `vertex_ai_pipeline.py` for specific configurations or tasks.
