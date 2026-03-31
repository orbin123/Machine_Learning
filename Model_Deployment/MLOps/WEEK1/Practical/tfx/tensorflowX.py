# ExampleGen
from tfx.components import CsvExampleGen

example_gen = CsvExampleGen(input_base='data/raw')

# StatisticsGen 
from tfx.components import StatisticsGen

statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

# SchemaGen
from tfx.components import SchemaGen 
schema_gen = SchemaGen(statistics_gen.outputs['statistics'])

# ExampleValidator
from tfx.components import ExampleValidator 

example_validator = ExampleValidator(
    statistics = statistics_gen.outputs['statistics'],
    schema=schema_gen.output['schema']
)

# Transform - Feature Engineering
from tfx.components import Transform

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'], 
    module_file = 'transform.py'
)

# Trainer
from tfx.components import Trainer
from tfx.proto import trainer_pb2

trainer = Trainer(
    module_file='model_module.py',
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=1000),
    eval_args=trainer_pb2.EvalArgs(num_steps=200)
)

# Evaluator 
from tfx.components import Evaluator
import tensorflow_model_analysis as tfma

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='target')],
    slicing_specs=[
        tfma.SlicingSpec(),                        # Overall
        tfma.SlicingSpec(feature_keys=['city']),   # Per city
    ],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='BinaryAccuracy'),
            tfma.MetricConfig(class_name='AUC'),
        ])
    ]
)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    eval_config=eval_config
)

# Pusher 
from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model/'
        )
    )
)