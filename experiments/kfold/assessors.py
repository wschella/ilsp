import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

import shared


def mnist() -> Model:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.mean_squared_error,
    )
    return model


def cifar10() -> Model:
    base = shared.baselines.cifar10.model(compiled=False)
    model = base
    return model
