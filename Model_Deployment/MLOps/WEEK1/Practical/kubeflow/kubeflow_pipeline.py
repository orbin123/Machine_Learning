import kfp 
from kfp import dsl

@dsl.component
def load_data() -> str:
    return "data/dataset.csv"

@dsl.component
def train_model(data_path: str) -> str: 
    return "/models/models.pkl"

@dsl.component
def evaluate_model(model_path: str) -> float:
    return 0.95

@dsl.pipeline(name="ML Training Pipeline")
def ml_pipeline():
    data_task = load_data()
    train_task = train_model(data_path=data_task.output)
    eval_task = evaluate_model(model_path=train_task.output)

kfp.compiler.Compiler().compile(ml_pipeline, "pipeline.yaml")