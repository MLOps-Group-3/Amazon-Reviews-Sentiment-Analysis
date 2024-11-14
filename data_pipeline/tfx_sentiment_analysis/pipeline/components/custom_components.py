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
    
    # Convert RaggedTensor to dense tensor and pad/truncate
    max_length = 128
    
    def ragged_to_dense(rt):
        values = rt.values
        row_splits = rt.row_splits
        dense = tf.RaggedTensor.from_row_splits(values, row_splits).to_tensor(default_value=0, shape=[None, max_length])
        return dense

    final_text = tf.map_fn(
        ragged_to_dense,
        encoded_text,
        fn_output_signature=tf.TensorSpec(shape=[None, max_length], dtype=tf.int64)
    )
    
    # Convert labels to integers
    label = inputs['sentiment_label']
    label_vocab = tft.vocabulary(label, top_k=10)
    encoded_label = tft.apply_vocabulary(label, label_vocab)
    
    return {
        'encoded_text': tf.cast(final_text, tf.float32),
        'label': tf.cast(encoded_label, tf.int64)
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
        # Ensure 'encoded_text' is float32 and 'label' is int64
        x['encoded_text'] = tf.cast(x['encoded_text'], tf.float32)
        x['label'] = tf.cast(x['label'], tf.int64)
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
