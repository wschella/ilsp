from __future__ import annotations
from typing import *

import tensorflow as tf
import tensorflow.keras as keras

from assessors.core import TFModelDefinition
import assessors.utils.callbacks_extra as cbe


class SegmentDefault(TFModelDefinition):
    epochs: int = 20

    def name(self) -> str:
        return "segment_default"

    def definition(self) -> keras.Model:
        model = keras.models.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(7, activation='softmax'),
        ])

        model.compile(
            # optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
        )

        return model

    def preproces_input(self, entry) -> tf.Tensor:
        return entry

    def score(self, y_true, y_pred) -> float:
        return (tf.math.argmax(y_pred, axis=1) == y_true)[0]

    def get_fit_kwargs(self, model, dataset, **kwargs) -> Dict:
        return {
            "verbose": 0,
            "callbacks": [
                cbe.EpochMetricLogger(total_epochs=self.epochs, metric="val_accuracy")
            ]
        }

    def train_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.cache()\
            .shuffle(10000)\
            .batch(64)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.cache()\
            .batch(32)\
            .prefetch(tf.data.experimental.AUTOTUNE)


class SegmentAssessorDefault(SegmentDefault):
    def name(self) -> str:
        return "segment_assessor_default"

    def definition(self) -> keras.Model:
        model = keras.models.Sequential([
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            # optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy(name='accuracy')],
        )

        return model
