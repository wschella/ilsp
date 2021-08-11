import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model


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
