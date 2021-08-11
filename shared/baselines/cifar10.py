import tensorflow as tf
import tensorflow.keras as keras
from keras import Model

from shared.models.wide_resnet import wide_resnet


def model(
    compiled: bool = True,
    input_shape=(32, 32, 3),
    depth=28,
    width_multiplier=10,
    l2=1e-4,
    num_classes=10,
    **kwargs
) -> Model:

    model = wide_resnet(
        input_shape=input_shape,
        depth=depth,
        width_multiplier=width_multiplier,
        l2=l2,
        num_classes=num_classes,
        **kwargs
    )

    if compiled:
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )

    return model


def model_with(**kwargs):
    return lambda: model(**kwargs)
