from __future__ import annotations
from typing import *
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.models.model import TFModel
import assessors.utils.callbacks_extra as callbacks
from assessors.models.shared.wide_resnet import wide_resnet


class CIFAR10Default(TFModel):
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    # depth: int = 28
    depth = 16
    # width_multiplier: int = 10
    width_multiplier = 1
    l2: float = 1e-4
    num_classes: int = 10
    model: Optional[keras.Model] = None
    epochs: int = 10

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

    def train(self, dataset, validation) -> CIFAR10Default:
        # Try first to restore an entire saved model
        if (model := self.try_restore_saved()) is not None:
            self.model = model
            return self

        # Try now to restore from checkpoints
        model = self.definition()
        (ckpt_manager, epoch) = self.get_checkpoint(model)
        ckpt_callback = callbacks.EpochCheckpointManagerCallback(ckpt_manager, epoch)

        # Actually train
        model.fit(
            train_pipeline(dataset),
            epochs=self.epochs,
            validation_data=test_pipeline(validation),
            initial_epoch=int(epoch),
            callbacks=[ckpt_callback]
        )
        self.save(model)
        return self

    def evaluate(self, dataset):
        self.model.evaluate(test_pipeline(dataset))


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


# def cifar10(**kwargs) -> Model:
#     base: Model = shared.baselines.cifar10.model(compiled=False, **kwargs)

#     # Remove the last (classification) layer, and add a regression one instead.
#     outputs = keras.layers.Dense(1, name="regression")(base.layers[-2].output)
#     model: Model = Model(inputs=base.inputs, outputs=outputs)
#     model.compile(
#         optimizer=keras.optimizers.Adam(0.001),
#         loss=keras.losses.mean_squared_error,
#     )
#     return model


def train_pipeline(ds: TFDataset):
    return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .cache()\
        .shuffle(1000)\
        .batch(128)\
        .prefetch(tf.data.experimental.AUTOTUNE)


def test_pipeline(ds: TFDataset):
    return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .cache()\
        .batch(128)\
        .prefetch(tf.data.experimental.AUTOTUNE)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
