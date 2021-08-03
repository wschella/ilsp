import tensorflow_datasets as tfds
import tensorflow as tf

from shared.models.wide_resnet import wide_resnet


def main():
    # We should concat again.
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        # We explicitly want control over shuffling ourselves
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )

    print(ds_info)

    # Train pipeline
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=1000)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Test pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Build model
    model = wide_resnet(input_shape=(32, 32, 3), depth=28,
                        width_multiplier=10, l2=1e-4, num_classes=10)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":
    main()
