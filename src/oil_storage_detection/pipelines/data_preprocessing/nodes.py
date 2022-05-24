"""Data preprocessing pipeline."""

import logging
import os
import shutil
from ast import literal_eval
from glob import glob
import numpy as np

import pandas as pd
from google.protobuf import text_format
from object_detection.protos.string_int_label_map_pb2 import (
    StringIntLabelMap,
    StringIntLabelMapItem,
)
from sklearn.model_selection import train_test_split

# configure standard logger
log = logging.getLogger(__name__)


def generate_label_map(classes: list[str], start: int):
    """Generates label map used for determining what object to detect.

    Args:
        classes (list[str]): _description_
        start (int): _description_

    Returns:
        _type_: _description_
    """
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), "utf-8")
    return text


def preprocess_labels(annotations: pd.DataFrame):
    """Converts bounds to python objects.

    Args:
        annotations (pd.DataFrame): Annotations containing the labels along with the bounding boxes.
    """

    annotations[["image_id", "bounds"]].copy()["bounds"] = annotations["bounds"].apply(
        lambda x: literal_eval(x.rstrip("\r\n"))
    )
    log.info(f"Wrote labels with the following shape: {annotations.shape}")
    return annotations


def train_test_split_node(
    labels: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
):
    """_summary_

    Args:
        labels (pd.DataFrame): _description_
        train_ratio (float): _description_
        validation_ratio (float): _description_
        test_ratio (float): _description_
        random_state (int): _description_

    Returns:
        _type_: _description_
    """

    train_image_names, test_image_names = train_test_split(
        labels, test_size=1 - train_ratio, random_state=random_state
    )

    val_image_names, test_image_names = train_test_split(
        test_image_names,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state,
    )

    # copy images to model_input folder
    os.makedirs("./data/04_feature/images/train", exist_ok=True)
    log.info(f"Writing {train_image_names.shape[0]} trainings images... ")
    for image_name in train_image_names["image_id"]:
        shutil.copy(
            f"./data/01_raw/images/{image_name}.jpg",
            f"./data/04_feature/images/train/{image_name}.jpg",
        )
    os.makedirs("../data/04_feature/images/validation", exist_ok=True)
    log.info(f"Writing {val_image_names.shape[0]} validation images... ")
    for image_name in val_image_names["image_id"]:
        shutil.copy(
            f"./data/01_raw/images/{image_name}.jpg",
            f"./data/04_feature/images/validation/{image_name}.jpg",
        )
    os.makedirs("../data/04_feature/images/test", exist_ok=True)
    log.info(f"Writing {test_image_names.shape[0]} test images... ")
    for image_name in test_image_names["image_id"]:
        shutil.copy(
            f"./data/01_raw/images/{image_name}.jpg",
            f"./data/04_feature/images/test/{image_name}.jpg",
        )

    # log resulting dataset sizes
    log.info(
        f"The trainings dataset contains {len(train_image_names)} images. Thats {round(len(train_image_names)/labels.shape[0] *100)}%."
    )
    log.info(
        f"The test dataset contains {len(test_image_names)} images. Thats {round(len(test_image_names)/labels.shape[0] *100)}%."
    )
    log.info(
        f"The validation dataset contains {len(val_image_names)} images. Thats {round(len(val_image_names)/labels.shape[0] *100)}%."
    )
    return train_image_names, test_image_names, val_image_names
