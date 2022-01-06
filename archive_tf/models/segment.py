from __future__ import annotations
from typing import *
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import assessors.utils.callbacks_extra as cbe
import assessors.models._base_models as _base_models


class SegmentDefault(_base_models.TFTabularClassification):
    epochs: int = 15

    def name(self) -> str:
        return "segment_default"

    def definition(self) -> keras.Model:
        model = keras.models.Sequential([
            # keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(10, activation='relu', input_dim=19),
            keras.layers.Dense(7, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.007),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
        )

        return model

    def get_fit_kwargs(self, model, dataset, **kwargs) -> Dict:
        return {
            "verbose": 0,
            "callbacks": [
                cbe.EpochMetricLogger(total_epochs=self.epochs, metric="val_accuracy")
            ],
        }


class SegmentAssessorDefault(SegmentDefault):
    epochs = 15
    n_systems = 5

    def name(self) -> str:
        return "segment_assessor_default"

    # def definition(self) -> keras.Model:
    #     model = keras.models.Sequential([
    #         # keras.layers.Dense(20, activation='relu'),
    #         keras.layers.Dense(10, activation='relu', input_dim=19),
    #         keras.layers.Dense(1, activation='sigmoid'),
    #     ])

    #     model.compile(
    #         optimizer=keras.optimizers.Adam(0.004),
    #         loss=keras.losses.BinaryCrossentropy(),
    #         metrics=[keras.metrics.BinaryAccuracy(name='accuracy')],
    #     )

    #     return model

    def definition(self) -> keras.Model:
        inst = layers.Input(shape=(19,))
        syst_id = layers.Input(shape=(self.n_systems,))

        # inst = layers.Dense(10, activation='relu')(inst)
        # syst_id = layers.Dense(1, activation='sigmoid')(syst_id)
        merged = layers.Concatenate()([inst, syst_id])
        hidden = layers.Dense(10, activation='relu')(merged)
        output = layers.Dense(1, activation='sigmoid')(hidden)

        model = keras.Model(inputs=[inst, syst_id], outputs=output)

        model.compile(
            optimizer=keras.optimizers.Adam(0.007),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy(name='accuracy')],
        )

        return model

    def get_fit_kwargs(self, model, dataset, **kwargs) -> Dict:
        return {
            "callbacks": [
                keras.callbacks.EarlyStopping(
                    patience=2, monitor="val_loss", restore_best_weights=True),
            ],
            "class_weight": {
                # 0 score is a failure, we want to detect these
                # currently, systems get about 95% accuracy, so 1/20 is a failure
                # 0: math.log(20),
                0: math.log(5),
                1: 1,
            }
        }
