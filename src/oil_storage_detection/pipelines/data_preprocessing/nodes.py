"""Data preprocessing pipeline."""

from ast import literal_eval
from glob import glob
import logging

# configure standard logger
log = logging.getLogger(__name__)


import pandas as pd


def get_image_names():
    """Retrieves the images names.
    """
    image_names = pd.DataFrame(
        {
            "image_id": [
                filename.split("/")[-1].split(".")[0]
                for filename in glob("./data/01_raw/images/*.jpg")
            ]
        }
    )
    log.info(f"Wrote image_names with the following shape: {image_names.shape}")
    return image_names


def preprocess_labels(annotations: pd.DataFrame):
    """Converts bounds to python objects.

    Args:
        annotations (pd.DataFrame): Annotations containing the labels along with the bounding boxes.
    """
    annotations["bounds"] = annotations["bounds"].apply(
        lambda x: literal_eval(x.rstrip("\r\n"))
    )[["image_id", "bounds"]]
    log.info(f"Wrote labels with the following shape: {annotations.shape}")
    return annotations
