"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from oil_storage_detection.pipelines import data_preprocessing as dp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    # Instantiate pipelines
    data_preprocessing_pipeline = dp.create_pipeline()

    return {
        "__default__": pipeline([data_preprocessing_pipeline]),
        "dp": data_preprocessing_pipeline,
    }
