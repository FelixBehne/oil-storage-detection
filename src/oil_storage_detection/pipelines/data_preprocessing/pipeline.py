"""Module for creating the data preprocessing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_image_names, preprocess_labels


def create_pipeline() -> Pipeline:
    """Creates the data preprocessing pipeline.

    Returns:
        Pipeline: Data Preprocessing Pipeline.
    """
    return pipeline(
        [
            node(
                func=get_image_names,
                inputs=None,
                outputs="image_names",
                name="get_image_names_node",
            ),
            node(
                func=preprocess_labels,
                inputs="annotations",
                outputs="labels",
                name="preprocess_labels_node",
            ),
        ]
    )
