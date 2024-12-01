# Data Preparation and Model Training Workflow

This section outlines the data preparation, model training, hyperparameter optimization, and evaluation process.

## 1. Data Preparation (`prepare_data.py`)

The `prepare_data.py` module handles raw data preparation for model training and evaluation. It performs the following tasks:

- **Prepares Raw Data**: Cleans and normalizes the input dataset.
- **Splits Data**: Splits the data into training, validation, and test sets.
- **Balances Data**: Balances the dataset by oversampling underrepresented categories to ensure fair model training.

### Functions
- `prepare_data()`
  - Prepares the raw data by cleaning, normalizing, and splitting it into training and test datasets.

- `balance_data(df, column="main_category", min_samples=50)`
  - Balances the dataset by oversampling underrepresented categories in the specified column (e.g., `main_category`) to meet a minimum sample threshold.
  - **Args**:
    - `df (pd.DataFrame)`: The input dataset.
    - `column (str)`: The column to be used for data slicing (default is `main_category`).
    - `min_samples (int)`: Minimum number of samples required for each category (default is 50).
  - **Returns**:
    - A balanced dataframe with over-sampled categories as necessary.

- `split_and_save_data(data_path, output_dir)`
  - Splits the data using the `split_data_by_timestamp()` function and saves the split data into separate training, validation, and test sets.
  - **Args**:
    - `data_path (str)`: Path to the raw data (e.g., CSV file).
    - `output_dir (str)`: Directory to save the pickled split data objects.

---

## 2. Hyperparameter Optimization using Optuna (`experiment_runner_optuna.py`)

The `experiment_runner_optuna.py` module handles the hyperparameter tuning of the sentiment model using the Optuna framework, with model tracking via `mlflow`.

### Functions
- `run_optuna_experiment()`
  - Runs hyperparameter optimization using Optuna.
  - Logs parameters, model performance, and artifacts using `mlflow`.
  - Saves the best hyperparameters based on performance metrics (e.g., F1 score).

- `objective(trial)`
  - Defines the optimization objective for Optuna.
  - Searches through the hyperparameter space and evaluates performance metrics for each set of hyperparameters.
  - **Args**:
    - `trial (optuna.trial)`: A trial object from the Optuna optimization framework.
  - **Returns**:
    - The evaluation score (e.g., F1 score) for the current hyperparameter set.

---
## 3. Final Model Training and Saving (`train_save_final_model.py`)

This module trains the sentiment analysis model using the best hyperparameters obtained from the hyperparameter optimization process and saves the trained model for deployment.

### Functions

- `train_and_save_final_model(hyperparameters, data_path, model_save_path)`
  - Trains the final model using the specified hyperparameters and training/validation datasets.
  - Saves the model's state dictionary locally for future deployment.
  - Pushes the saved model artifact to a Google Cloud Storage (GCS) bucket.
  - **Args**:
    - `hyperparameters (dict)`: The best hyperparameters for training (e.g., model name, learning rate, batch size, etc.).
    - `data_path (str)`: Path to the directory containing train/validation data in pickled format.
    - `model_save_path (str)`: Path to save the trained model.

- `upload_to_gcs(bucket_name, source_file_name, destination_blob_name)`
  - Uploads the saved model to a specified GCS bucket for cloud-based storage and accessibility.
  - **Args**:
    - `bucket_name (str)`: Name of the GCS bucket.
    - `source_file_name (str)`: Path to the local model file to be uploaded.
    - `destination_blob_name (str)`: Path in the GCS bucket to store the uploaded model file.

### Key Processes
- **Hyperparameter Loading**: Reads the best hyperparameters from a `best_hyperparameters.json` file.
- **Dataset Loading**: Loads the training and validation datasets from pickled files.
- **Model Initialization**: Initializes the BERT or RoBERTa model and tokenizer.
- **Training**: Trains the model with the training dataset and validates on the validation dataset.
- **Model Saving**: Saves the trained model to a specified local directory.
- **Model Uploading**: Optionally uploads the model to a Google Cloud Storage bucket for deployment.
---
## 4. Model Evaluation (`evaluate_model.py`)

This module evaluates the performance of the trained sentiment analysis model on a test dataset, calculating various metrics like accuracy, precision, recall, and F1 score. It also generates a detailed classification report.

### Functions

- `load_test_data(data_save_path)`
  - Loads the test dataset from a pickled file and verifies the presence of the `label` column.
  - **Args**:
    - `data_save_path (str)`: Path to the directory containing the test dataset in pickled format.
  - **Returns**:
    - A pandas DataFrame containing the test dataset.

- `load_label_encoder(data_save_path)`
  - Loads the label encoder from a pickled file for mapping labels to class names.
  - **Args**:
    - `data_save_path (str)`: Path to the directory containing the label encoder file.
  - **Returns**:
    - A label encoder object.

- `initialize_model_and_tokenizer(model_name)`
  - Initializes the specified model (BERT or RoBERTa) and its tokenizer.
  - **Args**:
    - `model_name (str)`: Name of the model to initialize (`"BERT"` or `"RoBERTa"`).
  - **Returns**:
    - The model initialization function and tokenizer.

- `evaluate_model(model, test_dataset, label_encoder)`
  - Evaluates the trained model on the test dataset and generates a classification report.
  - **Args**:
    - `model`: The trained model.
    - `test_dataset (SentimentDataset)`: The test dataset for evaluation.
    - `label_encoder`: The label encoder for mapping predictions back to class names.
  - **Returns**:
    - Overall metrics: accuracy, precision, recall, and F1 score.

- `load_hyperparameters(file_path)`
  - Loads the hyperparameters from a JSON file.
  - **Args**:
    - `file_path (str)`: Path to the hyperparameters JSON file.
  - **Returns**:
    - A dictionary of hyperparameters.

### Key Processes
- **Test Data Loading**: Reads the test dataset from a pickled file and validates its structure.
- **Model Initialization**: Initializes the trained model and loads its state dictionary.
- **Dataset Conversion**: Converts the test DataFrame to a `SentimentDataset` object for evaluation.
- **Evaluation**: Computes key metrics (accuracy, precision, recall, and F1 score) and generates a classification report.

---
## 5. Slice-Based Evaluation and Bias Detection (`slice_evaluation.py`)

This module evaluates the performance of the trained sentiment analysis model across various data slices, helping to identify potential biases in the model. It calculates metrics for slices of the dataset (e.g., based on `year` or `main_category`) and for the entire dataset.

### Functions

- `initialize_model_and_tokenizer(model_name)`
  - Initializes the specified model (BERT or RoBERTa) and its tokenizer.
  - **Args**:
    - `model_name (str)`: Name of the model to initialize (`"BERT"` or `"RoBERTa"`).
  - **Returns**:
    - The model initialization function and tokenizer.

- `load_hyperparameters(file_path)`
  - Loads hyperparameters from a JSON file.
  - **Args**:
    - `file_path (str)`: Path to the hyperparameters JSON file.
  - **Returns**:
    - A dictionary containing the hyperparameters.

- `evaluate_slices(data, model, tokenizer, data_path)`
  - Evaluates the model on specific slices of the dataset defined by columns like `year` or `main_category`, and on the full dataset.
  - **Args**:
    - `data (pd.DataFrame)`: The dataset to evaluate.
    - `model`: The trained model.
    - `tokenizer`: Tokenizer corresponding to the model.
    - `data_path (str)`: Path to save slice-based evaluation metrics.
  - **Returns**:
    - A DataFrame containing metrics for each slice and the entire dataset.

- `evaluate_dataset(data, model, tokenizer)`
  - Computes evaluation metrics (accuracy, precision, recall, and F1 score) for a given dataset.
  - **Args**:
    - `data (pd.DataFrame)`: The dataset to evaluate.
    - `model`: The trained model.
    - `tokenizer`: Tokenizer corresponding to the model.
  - **Returns**:
    - A dictionary containing metrics for the dataset.

### Key Processes

1. **Slice-Based Evaluation**:
   - The dataset is sliced into subsets based on predefined columns (e.g., `year`, `main_category`).
   - Metrics such as accuracy, precision, recall, and F1 score are computed for each slice.

2. **Full Dataset Evaluation**:
   - Metrics are computed for the entire dataset as a baseline for comparison.

3. **Bias Detection**:
   - The performance metrics for each slice are compared to identify potential biases in the model.
   - Slices with significantly lower performance metrics (e.g., low F1 scores for specific categories) indicate potential bias.

4. **Metrics Storage**:
   - Slice-based and full-dataset metrics are saved to a CSV file (`slice_metrics.csv`) for further analysis.

---
## 6. Bias Detection (`bias_detection.py`)

This module identifies potential biases in the model by analyzing performance metrics for different slices of the dataset. It flags slices with significantly lower F1 scores compared to the full dataset, helping to ensure fairness and robustness in the model's predictions.

### Functions

- `detect_bias(slice_metrics_path)`
  - Analyzes the slice-based evaluation metrics and identifies slices where the model may be underperforming relative to the full dataset.
  - **Args**:
    - `slice_metrics_path (str)`: Path to the CSV file containing slice-based metrics.
  - **Process**:
    1. Loads the slice metrics from the specified CSV file.
    2. Extracts metrics for the full dataset as a baseline.
    3. Defines thresholds for identifying bias:
       - Minimum samples: 10% of the full dataset's sample size.
       - F1 score threshold: 90% of the full dataset's F1 score.
    4. Filters and flags slices meeting these criteria as potentially biased.
  - **Outputs**:
    - Logs potential biases to the console and log file.
    - Provides insights into slices where the model may require additional analysis or retraining.

### Key Processes

1. **Full Dataset Baseline**:
   - Extracts the metrics of the full dataset from the slice metrics CSV to serve as a performance benchmark.

2. **Threshold Definition**:
   - Sets thresholds based on the full dataset's metrics:
     - **Minimum Samples Threshold**: Slices must have at least 10% of the full dataset's sample size.
     - **F1 Score Threshold**: Slices with F1 scores below 90% of the full dataset's F1 score are flagged.

3. **Bias Detection**:
   - Compares each slice's metrics to the thresholds and flags slices that meet the bias criteria.

4. **Results Logging**:
   - Logs the identified slices with potential bias, including their column, value, sample size, and F1 score.
---

