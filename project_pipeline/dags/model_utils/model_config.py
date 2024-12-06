import os

GCP_PROJECT="amazonreviewssentimentanalysis"
BUCKET_NAME="model-deployment-from-airflow"
GCP_REGION="us-central1"

DATA_PATH = f"gs://{BUCKET_NAME}/input/labeled_data.csv"
OUTPUT_DIR = f"gs://{BUCKET_NAME}/output/data/"
CODE_BUCKET_PATH = f"gs://{BUCKET_NAME}/code"
SOURCE_CODE = f"gs://{BUCKET_NAME}/code/src"
SLICE_METRIC_PATH = f"gs://{BUCKET_NAME}/output/metrics"
MODEL_SAVE_PATH = f"gs://{BUCKET_NAME}/output/models/train_job/final_model.pth"
MODEL_ARCHIVE_PATH = f"gs://{BUCKET_NAME}/output/models/archive/"
VERSION = 1
# APP_NAME = "review_sentiment_bert_model"
APP_NAME="review_sentiment_bert_model"

MODEL_DISPLAY_NAME = f"{APP_NAME}-v{VERSION}"
MODEL_DESCRIPTION = (
    "PyTorch serve deployment model for Amazon reviews classification"
)

health_route = "/ping"
predict_route = f"/predictions/{APP_NAME}"
serving_container_ports = [7080]

PROJECT_ID = GCP_PROJECT
DOCKER_IMAGE_NAME = f"pytorch_predict_{APP_NAME}"
CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/{DOCKER_IMAGE_NAME}"

GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY", "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json")
AIRFLOW_LOCAL = "/opt/airflow/dags/model_utils/src/"