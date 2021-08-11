import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

import shared


def mnist() -> Model:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, name="regression")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.mean_squared_error,
    )
    return model


def cifar10(**kwargs) -> Model:
    base: Model = shared.baselines.cifar10.model(compiled=False, **kwargs)

    # Remove the last (classification) layer, and add a regression one instead.
    outputs = keras.layers.Dense(1, name="regression")(base.layers[-2].output)
    model: Model = Model(inputs=base.inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.mean_squared_error,
    )
    return model
