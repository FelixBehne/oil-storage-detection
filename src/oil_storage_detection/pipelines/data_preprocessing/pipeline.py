"""Module for creating the data preprocessing pipeline."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    convert_to_tfrecords,
    generate_label_map,
    preprocess_labels,
    train_test_split_node,
)


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
            node(
                func=convert_to_tfrecords,
                inputs=[
                    "train_image_names",
                    "test_image_names",
                    "validation_image_names",
                    "params:image_width",
                    "params:image_height",
                ],
                outputs=["train_records", "test_records", "validation_records",],
                name="convert_to_tfrecords_node",
            ),
        ]
    )
