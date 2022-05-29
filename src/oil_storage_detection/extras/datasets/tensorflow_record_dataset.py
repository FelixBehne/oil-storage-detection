"""TensorflowRecordDataset."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np
import tensorflow.compat.v1 as tf
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class TensorflowRecordDataset(AbstractDataSet):
    """``TensorflowRecordDataset`` loads / saves a list of tfrecords to a
    tfrecord file using Tensorflow.

    Example:
    ::

        >>> TensorflowRecordDataset(filepath='/tfrecords/train.tfrecord')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of TensorflowRecordDataset to load / save
        tfrecord data for given filepath.

        Args:
            filepath: The location of the tfrecord file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        raw_dataset = tf.data.TFRecordDataset(load_path)
        return raw_dataset

    def _save(self, data: list[tf.train.Example]) -> None:
        """Saves image data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)

        # Write the `tf.train.Example` observations to the file.
        with tf.io.TFRecordWriter(save_path) as writer:
            for record in data:
                writer.write(record.SerializeToString())
            writer.close()

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
