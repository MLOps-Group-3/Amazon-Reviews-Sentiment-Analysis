
FROM pytorch/torchserve:latest-cpu

# Install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers

RUN ls -l

# Switch to the model-server user
USER model-server

# Set the model directory path
WORKDIR /home/model-server/

# Copy model artifacts, custom handler, and utilities
COPY ./bert-sent-model /home/model-server/bert-sent-model/
COPY ./custom_handler.py /home/model-server/custom_handler.py
# COPY ./data_loader.py /home/model-server/data_loader.py
# COPY ./bert_model.py /home/model-server/bert_model.py
COPY ./utils /home/model-server/utils/
COPY ./index_to_name.json /home/model-server/

# Verify the contents of the server directory
RUN ls -l /home/model-server

# Switch to root user for configuration
USER root

# Create TorchServe configuration file
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
RUN printf "\ndisable_token_authorization=true" >> /home/model-server/config.properties

# Change permissions for the model-server user
RUN chown -R model-server:model-server /home/model-server

# Switch back to the model-server user
USER model-server

# Expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# Create the TorchServe model archive
RUN torch-model-archiver -f     --model-name=review_sentiment_bert_model     --version 1.0     --serialized-file /home/model-server/bert-sent-model/final_model.pth     --handler /home/model-server/custom_handler.py     --extra-files "/home/model-server/utils/bert_model.py,/home/model-server/utils/data_loader.py,/home/model-server/utils/roberta_model.py,/home/model-server/index_to_name.json"     --export-path /home/model-server/model-store

# Start TorchServe
CMD ["torchserve",      "--start",      "--ts-config=/home/model-server/config.properties",      "--models",      "review_sentiment_bert_model=review_sentiment_bert_model.mar",      "--model-store",      "/home/model-server/model-store"]
