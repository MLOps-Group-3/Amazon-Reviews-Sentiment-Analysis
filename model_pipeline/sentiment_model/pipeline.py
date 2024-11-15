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

# Flag file paths to prevent infinite retriggering
FLAG_PATH_F1 = os.path.join(SCRIPT_DIR, ".pipeline_f1_retriggered")
FLAG_PATH_BIAS = os.path.join(SCRIPT_DIR, ".pipeline_bias_retriggered")

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

# Function to restart the pipeline from a specific step
def restart_pipeline_from_experiment_runner():
    """
    Retriggers the pipeline from the Experiment Runner step.
    """
    logging.info("Retriggering the pipeline from the Experiment Runner due to low F1 score.")
    experiment_runner_script = os.path.join(SCRIPT_DIR, "experiment_runner_optuna.py")
    pipeline_script = os.path.join(SCRIPT_DIR, "pipeline.py")
    try:
        # Run experiment runner
        process = subprocess.Popen(
            ["python", experiment_runner_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            logging.info("Experiment Runner retriggered successfully. Continuing pipeline.")
            print(stdout)
            # Continue the pipeline after Experiment Runner
            process_pipeline(pipeline_script, from_step="train_save.py")
        else:
            logging.error("Experiment Runner retriggering failed.")
            print(stderr)
    except Exception as e:
        logging.error(f"Error while retriggering Experiment Runner: {str(e)}")

# Function to continue the pipeline from a specific step
def process_pipeline(pipeline_script, from_step):
    """
    Run subsequent steps of the pipeline from a specified script.
    """
    steps = ["train_save.py", "evaluate_model.py", "evaluate_model_slices.py", "bias_detect.py"]
    for step in steps[steps.index(from_step):]:
        _, success = run_script(step)
        if not success:
            logging.error(f"Pipeline terminated due to failure in {step}.")
            return

# Master pipeline flow
def main():
    # Check for retrigger flags to prevent infinite loops
    if os.path.exists(FLAG_PATH_F1):
        logging.info("Pipeline retrigger due to low F1 score detected. Exiting to prevent infinite loop.")
        os.remove(FLAG_PATH_F1)  # Remove the flag to allow future retriggers
        return
    if os.path.exists(FLAG_PATH_BIAS):
        logging.info("Pipeline retrigger due to bias handling detected. Exiting to prevent infinite loop.")
        os.remove(FLAG_PATH_BIAS)  # Remove the flag to allow future retriggers
        return

    logging.info("Pipeline started.")
    
    # Step 1: Prepare Data
    _, success = run_script("src/prepare_data.py")
    if not success:
        logging.error("Pipeline terminated due to failure in prepare_data.py.")
        return
    
    # Step 2: Experiment Runner
    _, success = run_script("src/experiment_runner_optuna.py")
    if not success:
        logging.error("Pipeline terminated due to failure in experiment_runner_optuna.py.")
        return
    
    # Step 3: Train and Save
    _, success = run_script("src/train_save.py")
    if not success:
        logging.error("Pipeline terminated due to failure in train_save.py.")
        return
    
    # Step 4: Evaluate Model
    output, success = run_script("src/evaluate_model.py")
    if not success:
        logging.error("Pipeline terminated due to failure in evaluate_model.py.")
        return

    # Extract F1 score from logs
    f1_score = extract_f1_score(output)
    if f1_score is None:
        logging.error("Failed to extract F1 score from evaluate_model.py logs.")
    else:
        logging.info(f"Model test F1 score: {f1_score}")

    # Retrigger pipeline if F1 score is below threshold
    if f1_score < 0.6:
        logging.warning(f"F1 score below threshold (0.6). Current F1 score: {f1_score}. Retriggering Experiment Runner.")
        with open(FLAG_PATH_F1, "w") as f:
            f.write("Pipeline retriggered due to low F1 score.")
        restart_pipeline_from_experiment_runner()
        return

    # Step 5: Evaluate Model Slices
    _, success = run_script("src/evaluate_model_slices.py")
    if not success:
        logging.error("Pipeline terminated due to failure in evaluate_model_slices.py.")
        return

    # Step 6: Detect Bias and Handle It
    _, success = run_script("src/bias_detect.py")
    if not success:
        logging.error("Pipeline terminated due to failure in bias_detect.py.")
        return

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
