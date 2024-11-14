# pipeline/components/custom_components.py

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio
from typing import List, Text
from tensorflow_metadata.proto.v0 import schema_pb2

def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""
    # Assuming 'text' is the main feature for sentiment analysis
    text = inputs['text']
    
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
    
    # Pad or truncate sequences to a fixed length
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_text, maxlen=100, padding='post', truncating='post')
    
    # Convert labels to integers
    label = tft.compute_and_apply_vocabulary(inputs['sentiment_label'])
    
    return {
        'encoded_text': encoded_text,
        'label': label
    }

def _input_fn(file_pattern: List[Text],
              data_accessor: tfxio.TFXIOProvider,
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:
    """Generates features and labels for training or evaluation."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        schema=schema,
        batch_size=batch_size,
        shuffle=True).repeat()

def model_fn():
    """Define a simple Keras model."""
    inputs = tf.keras.Input(shape=(100,), dtype=tf.float32, name='encoded_text')
    x = tf.keras.layers.Embedding(10000, 64, input_length=100)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)  # Assuming 3 sentiment classes
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    schema = tfxio.TensorAdapter.SchemaCoder.load_schema(fn_args.schema_file)
    
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=fn_args.train_steps)
    
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=fn_args.eval_steps)
    
    model = model_fn()
    
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)
    
    model.save(fn_args.serving_model_dir, save_format='tf')

def get_eval_config():
    """Returns the evaluation configuration."""
    return tf.estimator.EvalConfig(
        model_dir=None,
        num_steps=1000,
        metrics_fn=lambda features, predictions: {
            'accuracy': tf.keras.metrics.Accuracy()
        }
    )
