# Data Pipelines in Airflow

This repository contains Airflow DAGs designed to handle multiple stages of data preprocessing, validation, and analytics for datasets. Below are the details of each DAG and their respective tasks.


# Data Pipeline Docker Setup

This setup guide walks you through building and running the data pipeline environment using Docker, including a custom Airflow image and Docker Compose for service management.

## Prerequisites
- **Docker Desktop** installed on your system.

## Setup Instructions

### 1. Build the Docker Image

To build the custom Docker image for the Airflow environment, run:

```bash
docker build -t custom-airflow:latest .
```

This will create an image based on the `Dockerfile` provided, which includes all necessary dependencies and directory structure for running the pipeline.

### 2. Start the Docker Compose Services

The `docker-compose.yaml` file configures the services, including Airflow, PostgreSQL, and Redis. Ensure environment variables for SMTP and Airflow user configurations are either set in your environment or in a `.env` file.

Setting up airflow service, run:

```bash
docker compose up airflow-init
```

This command will:
- Start Airflow services (web server, scheduler, worker, and triggerer).


To start all services, run:

```bash
docker compose up -d
```

This command will:
- Restart Airflow services (web server, scheduler, worker, and triggerer).
- Set up **Redis** as the message broker.
- Set up **PostgreSQL** as the Airflow database backend.

### 3. Access the Airflow Web Interface

Once all services are running, you can access the Airflow web interface at [http://localhost:8080](http://localhost:8080).

The default credentials for Airflow are:
- **Username**: `airflow`
- **Password**: `airflow`

### 4. Verify Setup

After logging in, you should see the DAGs located in the `data_pipeline/dags` folder displayed in the Airflow interface.

## DAGs Overview

1. **Data Acquisition**: This DAG handles the extraction and ingestion of Amazon review data.
2. **Data Sampling**: This DAG samples the data across various categories.
3. **Data Validation**: This DAG validates the quality and structure of the data.
4. **Data Preprocessing**: This DAG cleans, labels, and extracts aspects from the data.

---

### DAG: Data Acquisition

**DAG ID**: `01_data_collection_dag`

This DAG performs data acquisition for Amazon review data. The DAG consists of two main tasks:

#### Task 1: Acquire Data (`acquire_data`)

- **Objective**: Acquires and ingests Amazon review data.
- **Process**:
  - Calls the `acquire_data` function from the `data_acquisition` module to perform the data extraction. 
  - Skips redownloading if file exists in `data/raw` directory.
  - This function handles the logic of retrieving data from the source and saving it locally.
- **Output**: Data is saved in a designated directory, ready for further processing.

![alt text](<01 DAG Gantt Chart.jpeg>)

<!-- ### Task Dependencies

- **Flow**:
  - In the **Data Acquisition** DAG, the flow starts with `acquire_data`, leading to either the success or failure email tasks based on the outcome.
   -->
---
### DAG: Data Sampling

**DAG ID**: `02_data_sampling_dag`

This DAG is designed to sample and process Amazon review data by category using Python and Pandas. It orchestrates a sequence of tasks that handle data loading, filtering, joining with metadata, and sampling by specified review categories.

#### Task 1: Sample Data by Category (`sample_data_{category}`)

- **Objective**: Sample data for each specified Amazon review category.
- **Details**:
  - For each category listed in the `CATEGORIES` configuration, a `PythonOperator` task is created to execute the `sample_category` function.
  - Each task loads, filters, joins, and samples review data using Pandas.
  - The `sample_category` function in `sampling.py`:
    - Reads and decompresses JSONL.GZ files containing reviews and metadata.
    - Filters reviews by date and removes unnecessary fields.
    - Joins review data with product metadata for enriched sampling.
    - Performs stratified sampling to ensure diverse representation across months and ratings.
  - The sampled data is then saved to CSV files, split by year (2018–2019 and 2020).
- **Output**: Saves sampled data as CSV files for each category in `TARGET_DIRECTORY_SAMPLED`, enabling further analysis by year.

#### Task 2: Concatenate Sampled Data (`concatenate_data`)

- **Objective**: Combine all sampled data CSV files across categories into unified datasets.
- **Details**:
  - This task reads the sampled data files from each category and consolidates them into two comprehensive CSVs: one for 2018–2019 and another for 2020.
  - The output CSV files are saved in the specified directory (`TARGET_DIRECTORY_SAMPLED`).
- **Output**: Produces `sampled_data_2018_2019.csv` and `sampled_data_2020.csv` containing concatenated review data for each time period.

#### Task 3: Trigger Data Validation DAG (`trigger_data_validation_dag`)

- **Objective**: Initiate the `03_data_validation_dag` after data concatenation is completed.
- **Details**:
  - This task triggers a separate DAG to validate the concatenated data, ensuring data quality and completeness.
  - It acts as a starting point for further data validation or processing steps in a separate pipeline.
- **Output**: Marks the completion of data sampling and concatenation, ensuring readiness for validation.

#### Configuration and Logging

- **Logging**: Logging is set up to capture both console and file logs. Detailed logs are saved in `/opt/airflow/logs`, as specified by the `LOG_DIRECTORY` variable, and each task's operations are recorded, including any errors.
- **Error Handling**: In case of task failure, an email alert is sent to `vallimeenaavellaiyan@gmail.com`.

![alt text](<02 DAG Gantt Chart.jpeg>)

---
### DAG: Data Validation

**DAG ID**: `03_data_validation_dag`

This DAG performs validation checks on the dataset to ensure data quality, integrity, and compliance with predefined standards. The DAG consists of several main tasks:

#### Task 1: Schema Validation (`schema_validation`)

- **Objective**: Validates the schema of the dataset against expected column types.
- **Process**:
  - Loads the dataset and checks if the schema matches the defined structure using the `validate_schema` function.
  - If validation fails, an error is raised, and the process stops.
- **Output**: Logs the status of schema validation.

#### Task 2: Range Check (`range_check`)

- **Objective**: Checks numerical columns for valid value ranges.
- **Process**:
  - Loads the dataset and applies the `check_range` function to identify any values that fall outside of acceptable limits.
- **Output**: Logs the rows that failed the range check and the overall status.

#### Task 3: Missing and Duplicates Check (`missing_duplicates`)

- **Objective**: Identifies any missing or duplicate entries in the dataset.
- **Process**:
  - Loads the dataset and uses the `find_missing_and_duplicates` function to determine if there are any missing values or duplicate rows.
- **Output**: Logs the indices of missing and duplicate rows along with the status.

#### Task 4: Privacy Compliance Check (`privacy_compliance`)

- **Objective**: Ensures that data complies with privacy regulations.
- **Process**:
  - Loads the dataset and applies the `check_data_privacy` function to identify any rows that may breach privacy guidelines.
- **Output**: Logs the rows that failed the privacy check and the status.

#### Task 5: Emoji Detection (`emoji_detection`)

- **Objective**: Detects and flags any emojis in the dataset.
- **Process**:
  - Loads the dataset and uses the `detect_emoji` function to identify any rows containing emojis.
- **Output**: Logs the indices of rows with emojis and the overall status.

#### Task 6: Anomaly Detection (`anomaly_detection`)

- **Objective**: Detects anomalies within the dataset.
- **Process**:
  - Loads the dataset and applies the `detect_anomalies` function to identify unusual patterns or outliers.
- **Output**: Logs the status of anomaly detection.

#### Task 7: Special Characters Detection (`special_characters_detection`)

- **Objective**: Checks for the presence of special characters in the dataset.
- **Process**:
  - Loads the dataset and applies the `check_only_special_characters` function to identify any rows with only special characters.
- **Output**: Logs the rows with special characters and the status.

#### Task 8: Review Length Check (`review_length_checker`)

- **Objective**: Validates the length of reviews and titles in the dataset.
- **Process**:
  - Loads the dataset and applies the `check_review_title_length` function to determine if any reviews or titles are too short or too long.
- **Output**: Logs the results of review length checks.

---

### Task Dependencies

- **Flow**:
  - All validation tasks are executed in parallel.
  - The final task, `save_results`, runs after all validation tasks complete successfully to aggregate results and save them to a CSV file.

![alt text](<03 DAG Gantt Chart.jpeg>)

### DAG: Data Preprocessing

**DAG ID**: `04_data_preprocessing_dag`

This DAG performs data cleaning, labeling, and aspect-based sentiment analysis on Amazon review data. The DAG consists of four main tasks:

#### Task 1: Data Cleaning (`data_cleaning`)

- **Objective**: Cleans the raw data, removes unwanted emojis based on a validation file, and saves the cleaned data.
- **Process**:
  - Loads raw data and a validation file with emoji indices to be removed.
  - Applies the `clean_amazon_reviews` function from the `data_cleaning_pandas` module, which removes unwanted content and emojis.
  - Saves the cleaned data to a specified file path.
- **Output**: `cleaned_data.csv` - the cleaned dataset without unwanted emojis.

#### Task 2.1: Data Labeling (`data_labeling`)

- **Objective**: Labels the cleaned data with overall sentiment.
- **Process**:
  - Loads the cleaned data file.
  - Applies the `apply_labelling` function from the `data_labeling` module, which generates sentiment labels based on the content.
  - Saves the labeled data.
- **Output**: `labeled_data.csv` - dataset labeled with overall sentiment.

#### Task 2.2: Aspect Extraction (`aspect_extraction`)

- **Objective**: Identifies and extracts specific aspects within the reviews, such as "delivery," "quality," and "cost."
- **Process**:
  - Loads the cleaned data file.
  - Uses synonyms for predefined aspects (e.g., "delivery" includes "shipping," "arrive") and applies the `tag_and_expand_aspects` function from the `aspect_extraction` module to tag relevant text.
  - Saves the aspect-extracted data.
- **Output**: `aspect_extracted_data.csv` - dataset with tagged aspects for further sentiment analysis.

#### Task 3: Aspect-Based Data Labeling (`data_labeling_aspect`)

- **Objective**: Applies sentiment labeling to specific aspects within the reviews.
- **Process**:
  - Loads the aspect-extracted data file.
  - Uses the `apply_vader_labeling` function from `aspect_data_labeling` to assign sentiment labels based on each identified aspect.
  - Saves the aspect-labeled data.
- **Output**: `labeled_aspect_data.csv` - dataset with aspect-specific sentiment labels.

---

### Task Dependencies

- **Flow**:
  - The `data_cleaning` task is executed first.
  - `data_cleaning` is followed by parallel tasks `aspect_extraction` and `data_labeling`.
  - Finally, both `aspect_extraction` and `data_labeling` tasks must complete before starting the `data_labeling_aspect` task.

![alt text](<04 DAG Gantt Chart.jpeg>)

## DVC Setup

This guide will help you set up DVC and configure it to pull data files from Google Cloud Storage for this project.

### Prerequisites

1. **Install DVC**  
   Install DVC by running the following command:
   ```bash
   pip install dvc
   ```

2. **Google Cloud Service Key**  
   Ensure you have a Google Cloud service account key JSON file, which is necessary for authentication.

### Configuration

1. **Configure DVC Remote**  
   Set the following configuration in your DVC config file (this may already be set):
   ```ini
   [core]
       remote = gcp_remote
   ['remote "gcp_remote"']
       url = gs://sentiment-analysis-amazon-reviews-data/dvcstore
   ```

2. **Export Google Cloud Credentials**  
   Export the path to your GCP service account key file. Replace `/path/to/your/google-service-key.json` with the actual path to your key:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/google-service-key.json'
   ```

### Pulling Data

Once you have set up the configuration and credentials, you can pull the data files from the remote storage by running:
```bash
dvc pull
```

### Expected Output

You should see an output similar to this in your terminal:
```plaintext
Collecting                                                                                                                               |17.0 [00:03, 5.32entry/s]
Fetching
Building workspace index                                                                                                                 |2.00 [00:00, 9.91entry/s]
Comparing indexes                                                                                                                        |20.0 [00:00, 44.3entry/s]
Applying changes                                                                                                                         |13.0 [00:00, 209file/s]
A       ../../../../data_pipeline/data/cleaned/
A       ../../../../data_pipeline/data/labeled/
A       ../../../../data_pipeline/data/validation/
A       ../../../../data_pipeline/data/sampled/
4 files added and 12 files fetched
```

### Directory Structure

After running `dvc pull`, the files will be saved in the `data_pipeline/data` folder with the following structure:
- `cleaned/`
- `labeled/`
- `validation/`
- `sampled/`

These folders contain the various stages of processed data ready for use in the project.

