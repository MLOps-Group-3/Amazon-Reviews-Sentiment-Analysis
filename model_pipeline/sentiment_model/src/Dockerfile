# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY app_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r app_requirements.txt

# Copy the rest of the application code
# Copy only necessary files
COPY app.py .
COPY utils/ ./utils/  


# Expose the port
EXPOSE 8080

# Set environment variables
ENV MODEL_NAME=BERT
ENV MODEL_FILE=BERT_final_model.pth
ENV GCS_BUCKET_NAME=model_storage_arsa
ENV GCS_MODEL_PATH=models/BERT_final_model.pth

# Start the app using gunicorn
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app

# docker build -t gcr.io/amazonreviewssentimentanalysis/pytorch-serving:latest .
# docker push gcr.io/amazonreviewssentimentanalysis/pytorch-serving:latest