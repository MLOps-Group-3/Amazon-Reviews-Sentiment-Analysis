# pipeline/components/custom_components.py

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from typing import List, Text
from tensorflow_metadata.proto.v0 import schema_pb2
from transformers import BertTokenizer, TFBertForSequenceClassification
import mlflow

def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""
    text = inputs['text']
    
    # Ensure text is a string
    text = tf.strings.as_string(text)
    
    # Convert to lowercase
    text = tf.strings.lower(text)
    
    # Remove punctuation
    text = tf.strings.regex_replace(text, r'[^\w\s]', '')
    
    # Tokenize (simple whitespace tokenization)
    tokens = tf.strings.split(text)
    
    # Create a vocabulary
    vocab = tft.vocabulary(tokens, top_k=10000)
    
    # Encode text using the vocabulary
    encoded_text = tft.apply_vocabulary(tokens, vocab)
    
    # Handle both sparse and dense tensors
    if isinstance(encoded_text, tf.SparseTensor):
        dense_text = tf.sparse.to_dense(encoded_text, default_value=0)
    else:
        dense_text = encoded_text
    
    # Pad or truncate sequences to a fixed length
    max_length = 128
    
    def pad_or_truncate(t):
        shape = tf.shape(t)
        paddings = [[0, tf.maximum(0, max_length - shape[0])]]
        padded = tf.pad(t, paddings, constant_values=0)
        return padded[:max_length]
    
    final_text = tf.map_fn(pad_or_truncate, dense_text, dtype=tf.int64)
    
    # Convert labels to integers
    label = tft.compute_and_apply_vocabulary(inputs['sentiment_label'])
    
    return {
        'encoded_text': tf.cast(final_text, tf.float32),
        'label': tf.cast(label, tf.int32)
    }

def _input_fn(file_pattern: List[Text],
              data_accessor: tf.data.Dataset,
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:
    """Generates features and labels for training or evaluation."""
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        schema=schema,
        batch_size=batch_size,
        shuffle=True)
    
    def _clean_data(x):
        # Ensure 'encoded_text' is float32 and 'label' is int32
        x['encoded_text'] = tf.cast(x['encoded_text'], tf.float32)
        x['label'] = tf.cast(x['label'], tf.int32)
        return x
    
    return dataset.map(_clean_data).repeat()

def model_fn():
    """Define a BERT model for sequence classification."""
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    
    inputs = tf.keras.Input(shape=(128,), dtype=tf.float32, name='encoded_text')
    outputs = model(inputs)[0]
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    schema = tf.io.gfile.GFile(fn_args.schema_file, "r").read()
    schema = schema_pb2.Schema.FromString(schema)
    
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        fn_args.train_batch_size)
    
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        fn_args.eval_batch_size)
    
    model = model_fn()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fn_args.custom_config['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=fn_args.num_epochs)
    
    model.save(fn_args.serving_model_dir, save_format='tf')
    
    # Log metrics with MLflow
    mlflow.set_tracking_uri(fn_args.custom_config.get('mlflow_tracking_uri'))
    mlflow.tensorflow.autolog()

def get_eval_config():
    """Returns the evaluation configuration."""
    return {
        'model_specs': [{'label_key': 'label'}],
        'slicing_specs': [{}],
        'metrics_specs': [{
            'metrics': [
                {'class_name': 'Accuracy'},
                {'class_name': 'AUC'},
                {'class_name': 'Precision'},
                {'class_name': 'Recall'},
            ]
        }]
    }
