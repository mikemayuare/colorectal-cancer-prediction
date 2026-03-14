import kfp.compiler as compiler
from kfp import dsl


@dsl.container_component
def data_processing_op():
    return dsl.ContainerSpec(
        image="mikemayuare/colorec-ml-prediction:latest",
        command=["uv", "run", "-m", "src.processing"],
    )


@dsl.container_component
def model_training_op():
    return dsl.ContainerSpec(
        image="mikemayuare/colorec-ml-prediction:latest",
        command=["uv", "run", "-m", "src.training"],
    )


@dsl.pipeline(
    name="colorectal-cancer-prediction",
    description="Pipeline to predict colorectal cancer risk",
)
def mlops_pipeline():
    data_processing_task = data_processing_op()
    model_training_task = model_training_op()
    model_training_task.after(data_processing_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline, package_path="pipeline.yaml"
    )
