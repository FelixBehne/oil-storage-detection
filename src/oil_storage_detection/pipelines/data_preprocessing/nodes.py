"""Data preprocessing pipeline."""

from ast import literal_eval
from glob import glob

import pandas as pd


def get_image_names():
    """Retrieves the images names.
    """
    image_list = [
        filename.split("/")[-1].split(".")[0]
        for filename in glob("./data/01_raw/images/*.jpg")
    ]
    return image_list


def preprocess_labels(annotations: pd.DataFrame):
    """Converts bounds to python object.

    Args:
        annotations (pd.DataFrame): Annotations containing the labels along with the bounding boxes.
    """
    annotations["bounds"] = [literal_eval(x.rstrip("\r\n")) for x in annotations]
    return annotations[["image_id", "bounds"]]
