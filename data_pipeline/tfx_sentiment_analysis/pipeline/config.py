# pipeline/config.py

# import os

# PIPELINE_NAME = "sentiment_analysis_pipeline"
# PIPELINE_ROOT = os.path.join(os.getcwd(), "pipeline_root")
# METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
# DATA_ROOT = os.path.join(os.getcwd(), "data")

# MODEL_NAME = "BERT"
# LEARNING_RATE = 2e-5
# BATCH_SIZE = 32
# NUM_EPOCHS = 3
# WEIGHT_DECAY = 0.01
# DROPOUT_RATE = 0.1

# pipeline/config.py

import os

PIPELINE_NAME = "sentiment_analysis_pipeline"
PIPELINE_ROOT = os.path.join(os.getcwd(), "pipeline_root")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
DATA_ROOT = os.path.join(os.getcwd(), "data")

MODEL_NAME = "BERT"
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.1
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Set your MLflow tracking URI here
MAX_LENGTH = 128  # Maximum length for text sequence padding
