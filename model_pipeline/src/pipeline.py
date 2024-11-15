import subprocess
import logging
import time
import re
import os

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to run a script and capture its output
def run_script(script_name):
    start_time = time.time()
    script_path = os.path.join(SCRIPT_DIR, script_name)  # Get the full path of the script
    try:
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Stream and log outputs
        stdout, stderr = process.communicate()
        end_time = time.time()

        if process.returncode == 0:
            # Success
            logging.info(f"SUCCESS: {script_name} completed in {end_time - start_time:.2f} seconds.")
            logging.info(f"Output:\n{stdout}")
            print(stdout)  # Print to terminal
            return stdout, True
        else:
            # Failure
            logging.error(f"FAILURE: {script_name} failed after {end_time - start_time:.2f} seconds.")
            logging.error(f"Error Output:\n{stderr}")
            print(stderr)  # Print to terminal
            return stderr, False
    except Exception as e:
        end_time = time.time()
        logging.error(f"ERROR: Failed to execute {script_name} after {end_time - start_time:.2f} seconds.")
        logging.error(str(e))
        print(str(e))  # Print to terminal
        return str(e), False

# Function to extract F1 score from logs
def extract_f1_score(log_output):
    # Assuming F1 score is logged in the format: "F1 Score: <value>"
    match = re.search(r"F1 Score:\s*([\d.]+)", log_output)
    if match:
        return float(match.group(1))
    return None

# Master pipeline flow
def main():
    logging.info("Pipeline started.")
    
    # Step 1: Prepare Data
    _, success = run_script("prepare_data.py")
    if not success:
        logging.error("Pipeline terminated due to failure in prepare_data.py.")
        return
    
    # Step 2: Experiment Runner
    _, success = run_script("experiment_runner_optuna.py")
    if not success:
        logging.error("Pipeline terminated due to failure in experiment_runner_optuna.py.")
        return
    
    # Step 3: Train and Save
    _, success = run_script("train_save.py")
    if not success:
        logging.error("Pipeline terminated due to failure in train_save.py.")
        return
    
    # Step 4: Evaluate Model
    output, success = run_script("evaluate_model.py")
    if not success:
        logging.error("Pipeline terminated due to failure in evaluate_model.py.")
        return

    # Extract F1 score from logs
    f1_score = extract_f1_score(output)
    if f1_score is None:
        logging.error("Failed to extract F1 score from evaluate_model.py logs.")
    else:
        logging.info(f"Model test F1 score: {f1_score}")

    # Step 5: Evaluate Model Slices
    _, success = run_script("evaluate_model_slices.py")
    if not success:
        logging.error("Pipeline terminated due to failure in evaluate_model_slices.py.")
        return
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
