from typing import List, Tuple
from dataclasses import dataclass
import logging

# We need to do this before importing TF
from shared.util.tf_logging import set_tf_loglevel
set_tf_loglevel(logging.WARN)  # nopep8

import click

import tensorflow.keras as keras
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow_datasets.core import DatasetInfo
from tensorflow.python.framework.ops import Tensor
from tensorflow_datasets.core.utils.read_config import ReadConfig

# Used for .flatten() on Tensors
import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()  # nopep8

import shared.util.callbacks_extra as cb_extra
import shared.util.dataset_extra as ds_extra


Dataset = tf.data.Dataset

LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


@dataclass
class BaseTrainArguments:
    retrain: bool


@dataclass
class AssessorTrainArguments:
    pass


@click.command()
@click.option('-r', '--retrain', is_flag=True, help='Wether the models should be retrained or try to reload from checkpoints (default)')
def main(retrain: bool):
    base_args = BaseTrainArguments(retrain)

    ds, ds_info = load_dataset()
    folds = ds_extra.k_folds(ds, 5)
    models = train_models(folds, ds_info, base_args)

    results: List[Dataset] = []
    for ((_train, test), model) in zip(folds, models):
        def to_assessor(x: Tensor, y_true):
            y_pred = model(x.reshape((1, 28, 28)))
            loss = LOSS(y_true, y_pred)
            return (x, loss)

        results.append(test.map(to_assessor))

    # Create the assessor dataset, which is the concatenation off all K test sets
    # where the label is the loss for the corresponding model
    ass_ds: tf.data.Dataset = ds_extra.concatenate_all(results)
    (ass_train_ds, ass_test_ds) = ds_extra.split_absolute(ass_ds, 60000)
    assert ass_ds.cardinality() == 70000

    assessor = build_assessor_model()
    assessor.fit(
        train_pipeline(ass_train_ds),
        epochs=6,
        validation_data=test_pipeline(ass_test_ds),
    )


def load_dataset() -> Tuple[Dataset, DatasetInfo]:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/load

    ds_info: DatasetInfo
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',

        # We need to concat later, but if we don't split,
        # we get a {"train": Dataset, "test": Dataset} object anyway.
        split=["train", "test"],

        # We explicitly want control over shuffling ourselves
        shuffle_files=False,

        # Returns tuple (img, label) instead of dict {'image': img, 'label': label}
        as_supervised=True,

        with_info=True,

        read_config=ReadConfig(
            try_autocache=False,
        )
    )

    ds: Dataset = ds_train.concatenate(ds_test)
    return (ds, ds_info)


def train_models(folds, ds_info, args: BaseTrainArguments) -> List[Model]:
    models: List[Model] = []
    for (i, (train, test)) in enumerate(folds):

        model = build_model()

        checkpoint_dir = f"models/kfold/mnist/{i}/"
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        checkpoint_callback = cb_extra.EpochCheckpointManagerCallback(manager)

        latest = manager.latest_checkpoint
        initial_epoch = 0
        if latest and not args.retrain:
            print("Using model checkpoint {}".format(latest))

            # We add .expect_partial(), because if a previous run completed,
            # and we consequently restore from the last checkpoint, no further
            # training is need, and we don't expect to use all variables.
            checkpoint.restore(latest).expect_partial()
            initial_epoch = int(checkpoint.save_counter)
        else:
            if args.retrain:
                print("Training from scratch due to --retrain")
            else:
                print("Training from scratch")

        model.fit(
            train_pipeline(train),
            epochs=6,
            validation_data=test_pipeline(test),
            callbacks=[checkpoint_callback],
            initial_epoch=initial_epoch
        )
        models.append(model)
    return models


def build_model() -> Model:
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=LOSS,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def build_assessor_model() -> Model:
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


def train_pipeline(ds_train: Dataset):
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train


def test_pipeline(ds_test: Dataset):
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_test


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":
    main()
