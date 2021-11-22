from __future__ import annotations
from typing import *

import tensorflow.keras as keras
import assessors.models._base_models as _base_models


class MNISTDefault(_base_models.TFImageClassification):
    epochs: int = 5

    def name(self) -> str:
        return "mnist_default"

    def definition(self) -> keras.Model:
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')],
        )

        return model


class MNISTAssessorDefault(MNISTDefault):
    epochs = 15

    def name(self) -> str:
        return "mnist_assessor_default"

    def definition(self) -> keras.Model:
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy(name='acc')],
        )

        return model

    def get_fit_kwargs(self, model, dataset, **kwargs) -> Dict:
        return {
            "class_weight": {
                # 0 score is a failure, we want to detect these
                # currently, systems get about 97% accuracy, so +- 1/33 is a failure
                0: 20,
                1: 1,
            }
        }

    def score(self, y_true, y_pred) -> float:
        raise NotImplementedError()
