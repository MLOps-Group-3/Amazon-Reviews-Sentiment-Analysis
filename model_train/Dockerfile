# Use the latest PyTorch GPU base image for Vertex AI
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install all Python packages listed in requirements.txt
RUN pip install -r requirements.txt

# Copy all files from the trainer directory into the container
COPY ./trainer /app/trainer

# Set up the entry point to invoke the trainer
ENTRYPOINT ["python", "-m", "trainer.training_script"]
