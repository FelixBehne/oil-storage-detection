"""Module for creating the data preprocessing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_label_map, preprocess_labels, train_test_split_node


def create_pipeline() -> Pipeline:
    """Creates the data preprocessing pipeline.

    Returns:
        Pipeline: Data Preprocessing Pipeline.
    """
    return pipeline(
        [
            node(
                func=generate_label_map,
                inputs=["params:classes", "params:start"],
                outputs="label_map",
                name="generate_label_map_node",
            ),
            node(
                func=preprocess_labels,
                inputs="annotations",
                outputs="labels",
                name="preprocess_labels_node",
            ),
            node(
                func=train_test_split_node,
                inputs=[
                    "labels",
                    "params:train_ratio",
                    "params:validation_ratio",
                    "params:test_ratio",
                    "params:random_state",
                ],
                outputs=[
                    "train_image_names",
                    "test_image_names",
                    "validation_image_names",
                ],
                name="train_test_split_node",
            ),
        ]
    )
