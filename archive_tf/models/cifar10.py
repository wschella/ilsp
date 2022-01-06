from __future__ import annotations
from typing import *

import tensorflow.keras as keras
from keras import layers

from assessors.models.architectures.wide_resnet import wide_resnet
import assessors.models._base_models as _base_models


class CIFAR10Default(_base_models.TFImageClassification):
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
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')],
        )
        return model


class CIFAR10AssessorDefault(CIFAR10Default):
    def name(self) -> str:
        return "cifar10_default_assessor"

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
            metrics=[keras.metrics.BinaryAccuracy(name='acc')],
        )
        return model

# ------------------ Wide Model ---------------------


class CIFAR10Wide(_base_models.TFImageClassification):
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
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')]
        )

        return definition


class CIFAR10AssessorWide(CIFAR10Wide):
    epochs = 25
    num_classes = 2

    def name(self) -> str:
        return "cifar10_wide_assessor"

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
            metrics=[keras.metrics.BinaryAccuracy(name='acc')]
        )

        return definition
