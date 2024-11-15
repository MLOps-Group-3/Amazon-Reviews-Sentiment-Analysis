# Sentiment Model Pipeline

This repository contains the model pipeline for training, evaluating, and monitoring a sentiment analysis model. The pipeline includes data preparation, hyperparameter tuning, model training, evaluation, and bias detection. It also integrates with MLflow for experiment tracking and supports retriggering for underperforming models.

## Project Structure

```plaintext
├── dags
│   ├── mlflow_tracking_dag.py      # Airflow DAG for MLflow experiment tracking
│   └── __pycache__
├── docker-compose.yaml             # Docker configuration for running the pipeline
├── pipeline.py                     # Master pipeline script
├── requirements.txt                # Python dependencies
└── src
    ├── app.py                      # Flask/Django app for model interaction (optional)
    ├── app_requirements.txt        # Additional requirements for the app
    ├── bert_output/                # Directory for BERT model logs
    ├── bias_detect.py              # Detect and mitigate model bias
    ├── config.py                   # Pipeline configuration settings
    ├── data/                       # Data directory
    ├── evaluate_model.py           # Evaluate the trained model
    ├── evaluate_model_slices.py    # Evaluate model performance on data slices
    ├── experiment_runner_optuna.py # Hyperparameter tuning with Optuna
    ├── experiment_runner.py        # General experiment runner
    ├── mlruns/                     # MLflow directory for experiment tracking
    ├── prepare_data.py             # Data preparation script
    ├── roberta_output/             # Directory for RoBERTa model logs
    ├── train_save.py               # Train and save the model
    └── utils/                      # Utility scripts
        ├── bert_model.py           # BERT model implementation
        ├── data_loader.py          # Data loading functions
        ├── roberta_model.py        # RoBERTa model implementation
```

## Features

- **Data Preparation**: Prepares raw data for training and evaluation, ensuring compatibility with the sentiment model.
- **Hyperparameter Tuning**: Uses Optuna for automated hyperparameter optimization.
- **Model Training and Evaluation**: Trains the sentiment model using BERT or RoBERTa and evaluates the model, logging metrics including the F1 score.
- **Bias Detection**: Identifies and mitigates biases in model predictions to ensure fairness.
- **F1 Score Monitoring**: Automatically extracts the F1 score and retriggers experiments if it falls below a threshold (default: 0.6).
- **Experiment Tracking**: Integrates with MLflow to track model training, metrics, and artifacts.
- **Flexible Pipeline Execution**: Supports restarting from any pipeline step after failure.
- **Containerized Workflow**: Dockerized setup for seamless development and deployment.

## Getting Started

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the following are configured:
   - MLflow for tracking experiments (`mlruns` directory).
   - Docker for running the pipeline in a containerized environment.

### Running the Pipeline

1. Run the entire pipeline:
   ```bash
   python pipeline.py
   ```
2. Restart the pipeline from a specific step: Modify `pipeline.py` to set the starting step and execute:
   ```bash
   python pipeline.py
   ```

### Configuration

Update `config.py` to adjust the pipeline's behavior, including:
- Model parameters
- Data paths

## Experiment Tracking with MLflow

- The `mlruns` directory contains logs and metrics for all experiments.
- Use the Airflow DAG in `dags/mlflow_tracking_dag.py` to automate MLflow tracking.

## Retriggering

The pipeline automatically retriggers under the following conditions:
1. **Low F1 Score**: Retriggers from hyperparameter tuning if the score is below 0.6.
2. **Bias Detection**: Retriggers from the bias mitigation step if biases are detected.

Flags are used to prevent infinite retrigger loops.

## Docker Setup

1. Build and start the pipeline with Docker Compose:
   ```bash
   docker-compose up --build
   ```
2. Modify the `docker-compose.yaml` file to adjust services as needed.

---

**Note**: Docker with MLflow integration was successful. However, Airflow with GPU compute setup was unsuccessful, and the pipeline is being transitioned to the cloud for scalability and better resource management.
