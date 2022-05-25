"""Data preprocessing pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import warnings
from ast import literal_eval
from pathlib import Path

import pandas as pd
from sqlalchemy import column
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection.protos.string_int_label_map_pb2 import (
    StringIntLabelMap,
    StringIntLabelMapItem,
)
from object_detection.utils import dataset_util
from sklearn.model_selection import train_test_split

# configure standard logger
log = logging.getLogger(__name__)


def _create_tf_records_from_files(
    image_path: str | Path,
    annotations: pd.DataFrame,
    image_width: int,
    image_height: int,
) -> list[tf.train.Example]:

    image_path = Path(image_path)
    image_filenames = os.listdir(image_path)

    tf_records = []

    for image_filename in image_filenames:
        with tf.gfile.GFile(image_path / image_filename, "rb") as fid:
            encoded_jpg = fid.read()

        image_width = image_width
        image_height = image_height

        filename = image_filename.encode("utf8")
        image_format = b"jpg"

        annotations_filtered = annotations[
            annotations["image_id"] == image_filename[:-4]
        ]

        xmins = annotations_filtered["bbox_x1"].values
        xmaxs = annotations_filtered["bbox_x2"].values
        ymins = annotations_filtered["bbox_y1"].values
        ymaxs = annotations_filtered["bbox_y2"].values

        class_name = "OST".encode("utf8")

        classes_text = [class_name for i in range(len(xmins))]
        classes = [1 for i in range(len(xmins))]

        tf_records.append(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image/height": dataset_util.int64_feature(image_height),
                        "image/width": dataset_util.int64_feature(image_width),
                        "image/filename": dataset_util.bytes_feature(filename),
                        "image/source_id": dataset_util.bytes_feature(filename),
                        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                        "image/format": dataset_util.bytes_feature(image_format),
                        "image/object/bbox/xmin": dataset_util.float_list_feature(
                            xmins
                        ),
                        "image/object/bbox/xmax": dataset_util.float_list_feature(
                            xmaxs
                        ),
                        "image/object/bbox/ymin": dataset_util.float_list_feature(
                            ymins
                        ),
                        "image/object/bbox/ymax": dataset_util.float_list_feature(
                            ymaxs
                        ),
                        "image/object/class/text": dataset_util.bytes_list_feature(
                            classes_text
                        ),
                        "image/object/class/label": dataset_util.int64_list_feature(
                            classes
                        ),
                    }
                )
            )
        )

    return tf_records


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

    annotations["bounds"] = annotations["bounds"].apply(
        lambda x: literal_eval(x.rstrip("\r\n"))
    )
    annotations = pd.concat(
        [
            annotations,
            pd.DataFrame(
                annotations["bounds"].tolist(),
                index=annotations.index,
                columns=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
            ),
        ],
        axis=1,
    ).drop(["bounds", "class"], axis=1)

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
        labels, train_size=train_ratio, random_state=random_state
    )

    val_image_names, test_image_names = train_test_split(
        test_image_names,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state,
    )

    # create dirs
    os.makedirs("./data/04_feature/images/test", exist_ok=True)
    os.makedirs("./data/04_feature/images/train", exist_ok=True)
    os.makedirs("./data/04_feature/images/validation", exist_ok=True)

    # copy images to model_input folder
    log.info(f"Writing {train_image_names.shape[0]} trainings images... ")
    for image_name in train_image_names["image_id"]:
        shutil.copy(
            f"./data/01_raw/images/{image_name}.jpg",
            f"./data/04_feature/images/train/{image_name}.jpg",
        )
    log.info(f"Writing {val_image_names.shape[0]} validation images... ")
    for image_name in val_image_names["image_id"]:
        shutil.copy(
            f"./data/01_raw/images/{image_name}.jpg",
            f"./data/04_feature/images/validation/{image_name}.jpg",
        )
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


def convert_to_tfrecords(
    train_image_names: pd.DataFrame,
    test_image_names: pd.DataFrame,
    val_image_names: pd.DataFrame,
    image_width: int,
    image_height: int,
):

    # create tfrecords from images and annotations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        tf_train_records = _create_tf_records_from_files(
            "./data/04_feature/images/train/",
            train_image_names,
            image_width,
            image_height,
        )
        tf_test_records = _create_tf_records_from_files(
            "./data/04_feature/images/test/",
            test_image_names,
            image_width,
            image_height,
        )
        tf_valid_records = _create_tf_records_from_files(
            "./data/04_feature/images/validation/",
            val_image_names,
            image_width,
            image_height,
        )

    return tf_train_records, tf_test_records, tf_valid_records
