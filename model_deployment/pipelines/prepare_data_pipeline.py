import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kfp.dsl import pipeline
from kfp import compiler
from google.cloud import aiplatform
from components.prepare_data_component import prepare_data_component

# Import configurations from config.py
from pipelines.config import PROJECT_ID, REGION, BUCKET_NAME, DATA_PATH, OUTPUT_DIR, PIPELINE_ROOT

@pipeline(
    name="prepare-data-pipeline",
    pipeline_root=PIPELINE_ROOT,  # Use PIPELINE_ROOT from config.py
)
def prepare_data_pipeline():
    prepare_data_component(
        bucket_name=BUCKET_NAME,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
    )

# Compile and submit the pipeline
if __name__ == "__main__":
    # Compile the pipeline
    pipeline_json = "prepare_data_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=prepare_data_pipeline,
        package_path=pipeline_json,
    )

    # Initialize AI Platform
    aiplatform.init(
        project=PROJECT_ID,  # Use PROJECT_ID from config.py
        location=REGION,  # Use REGION from config.py
    )

    # Submit the pipeline
    job = aiplatform.PipelineJob(
        display_name="prepare-data-pipeline",
        template_path=pipeline_json,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={},  # No parameters are needed as all values are from config.py
    )
    job.run(sync=True)  # Set `sync=False` to run asynchronously
