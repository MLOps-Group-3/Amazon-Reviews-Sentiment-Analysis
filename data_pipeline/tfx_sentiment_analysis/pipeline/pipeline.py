# pipeline/pipeline.py

import os
from typing import List
from absl import logging
from tfx import v1 as tfx
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from ml_metadata.proto import metadata_store_pb2

from pipeline.components import custom_components
from pipeline.config import (
    PIPELINE_NAME,
    PIPELINE_ROOT,
    METADATA_PATH,
    DATA_ROOT,
    MODEL_NAME,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    DROPOUT_RATE,
)

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    dropout_rate: float,
    enable_cache: bool,
    metadata_connection_config: metadata_store_pb2.ConnectionConfig,
    beam_pipeline_args: List[str],
) -> tfx.dsl.Pipeline:
    components = []

    # ExampleGen component
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)
    components.append(example_gen)

    # StatisticsGen component
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # SchemaGen component
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)
    components.append(schema_gen)

    # ExampleValidator component
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

    # Transform component
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=custom_components.preprocessing_fn)
    components.append(transform)

    # Trainer component
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath(custom_components.__file__),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=num_epochs),
        eval_args=tfx.proto.EvalArgs(num_steps=num_epochs),
        custom_config={
            'model_name': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'dropout_rate': dropout_rate,
        })
    components.append(trainer)

    # Resolver component
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)).with_id('latest_blessed_model_resolver')
    components.append(model_resolver)

    # Evaluator component
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=custom_components.get_eval_config())
    components.append(evaluator)

    # Pusher component
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=os.path.join(pipeline_root, 'pushed_models'))))
    components.append(pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )

def run_pipeline():
    tfx.orchestration.LocalDagRunner().run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            model_name=MODEL_NAME,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            dropout_rate=DROPOUT_RATE,
            enable_cache=True,
            metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
            beam_pipeline_args=[],
        )
    )

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run_pipeline()
