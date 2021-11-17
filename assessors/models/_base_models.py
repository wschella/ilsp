from __future__ import annotations
from abc import ABC
from typing import *

import tensorflow as tf

from assessors.core import TFModelDefinition


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label  # type: ignore


class TFTabularClassification(TFModelDefinition, ABC):
    """
    Base definition for Tensorflow image classification models, 
    providing pipelines and score functions shared by all.
    """

    def preproces_input(self, entry) -> tf.Tensor:
        return normalize_img(entry, None)[0]

    def score(self, y_true, y_pred) -> float:
        return (tf.math.argmax(y_pred, axis=1) == y_true)[0]

    def train_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .shuffle(10000)\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(256)\
            .prefetch(tf.data.experimental.AUTOTUNE)


class TFImageClassification(TFModelDefinition, ABC):
    """
    Base definition for Tensorflow image classification models, 
    providing pipelines and score functions shared by all.
    """

    def preproces_input(self, entry) -> tf.Tensor:
        return normalize_img(entry, None)[0]

    def score(self, y_true, y_pred) -> float:
        return (tf.math.argmax(y_pred, axis=1) == y_true)[0]

    def train_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .shuffle(1000)\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(256)\
            .prefetch(tf.data.experimental.AUTOTUNE)
