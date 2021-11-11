from __future__ import annotations
from typing import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.core import TFModelDefinition
from assessors.models.architectures.wide_resnet import wide_resnet


class CIFAR10Default(TFModelDefinition):
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    # depth: int = 28
    depth = 16
    # width_multiplier: int = 10
    width_multiplier = 1
    l2: float = 1e-4
    num_classes: int = 10
    model: Optional[keras.Model] = None
    epochs: int = 10

    def name(self) -> str:
        return "cifar10_default"

    def definition(self, compiled: bool = True):
        definition = wide_resnet(
            input_shape=self.input_shape,
            depth=self.depth,
            width_multiplier=self.width_multiplier,
            l2=self.l2,
            num_classes=self.num_classes,
        )

        definition.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )

        return definition

    def train_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .shuffle(1000)\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)


class CIFAR10AssessorProbabilistic(CIFAR10Default):
    epochs = 25
    num_classes = 2

    def definition(self, compiled: bool = True):
        definition = wide_resnet(
            input_shape=self.input_shape,
            depth=self.depth,
            width_multiplier=self.width_multiplier,
            l2=self.l2,
            num_classes=2,
        )

        definition.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()]
        )

        return definition


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
