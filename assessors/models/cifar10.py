from __future__ import annotations
from abc import ABC
from typing import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset
from keras import layers

from assessors.core import TFModelDefinition
from assessors.core.model import Restore, TrainedModel
from assessors.models.architectures.wide_resnet import wide_resnet


class CIFAR10Model(ABC):
    """
    Base configuration for CIFAR10 models, providing pipelines and score functions
    shared by all.
    """

    def score(self, y_true, prediction):
        return float(tf.math.argmax(prediction, axis=1) == y_true)

    def train_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .shuffle(1000)\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(256)\
            .prefetch(tf.data.experimental.AUTOTUNE)


class CIFAR10Default(CIFAR10Model, TFModelDefinition):
    epochs: int = 10

    def name(self) -> str:
        return "cifar10_default"

    def definition(self):
        model = keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        return model


class CIFAR10AssessorProbabilisticDefault(CIFAR10Default):
    def name(self) -> str:
        return "cifar10_assessor_probabilistic_default"

    def definition(self):
        model = keras.models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()],
        )
        return model

# ------------------ Wide Model ---------------------


class CIFAR10Wide(TFModelDefinition):
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
        return "cifar10_wide"

    def definition(self,):
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


class CIFAR10AssessorProbabilisticWide(CIFAR10Wide):
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

    def train(self, dataset, validation, restore: Restore) -> TrainedModel:
        return self._train(dataset, validation, restore)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
