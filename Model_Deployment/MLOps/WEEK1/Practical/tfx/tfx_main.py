import os
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Trainer,
    Pusher
)
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import (
    InteractiveContext
)

# --- Setup paths ---
PIPELINE_NAME = "simple_tfx_pipeline"
DATA_ROOT = os.path.join(os.getcwd(), "data")         # folder with CSV file
PIPELINE_ROOT = os.path.join(os.getcwd(), "pipeline_output")
SERVING_DIR = os.path.join(os.getcwd(), "serving_model")

# --- Create interactive context (for local/notebook runs) ---
context = InteractiveContext(pipeline_name=PIPELINE_NAME)

# --- Component 1: Ingest Data ---
example_gen = CsvExampleGen(input_base=DATA_ROOT)
context.run(example_gen)

# --- Component 2: Compute Statistics ---
statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
context.run(statistics_gen)

# --- Component 3: Infer Schema ---
schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
context.run(schema_gen)

# --- Component 4: Validate Data ---
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs["statistics"],
    schema=schema_gen.outputs["schema"]
)
context.run(example_validator)

# --- Component 5: Train Model ---
trainer = Trainer(
    module_file=os.path.join(os.getcwd(), "trainer_module.py"),  # see below
    examples=example_gen.outputs["examples"],
    schema=schema_gen.outputs["schema"],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=50)
)
context.run(trainer)

# --- Component 6: Push Model ---
pusher = Pusher(
    model=trainer.outputs["model"],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=SERVING_DIR
        )
    )
)
context.run(pusher)

print("Pipeline complete! Model saved to:", SERVING_DIR)