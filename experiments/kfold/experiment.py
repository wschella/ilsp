from typing import List, Tuple, Callable
from dataclasses import dataclass
from os import path
import logging

# We need to do this before importing TF
from shared.util.tf_logging import set_tf_loglevel
set_tf_loglevel(logging.WARN)  # nopep8


import click
import wandb

import tensorflow.keras as keras
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from wandb.keras import WandbCallback
from tensorflow.keras import Model
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow_datasets.core import DatasetInfo
from tensorflow.python.framework.ops import Tensor
from tensorflow_datasets.core.utils.read_config import ReadConfig
import tensorflow.python.ops.numpy_ops.np_config as np_config

import shared.util.callbacks_extra as cb_extra
import shared.util.dataset_extra as ds_extra
import shared

import experiments.kfold.assessors

# Used for .flatten() on Tensors
np_config.enable_numpy_behavior()  # nopep8

Dataset = tf.data.Dataset


@dataclass
class Config:
    epochs: int
    folds: int
    assessor_test_set_size: int
    model: Callable[[], Model]
    assessor: Callable[[], Model]


CONFIG = {
    'mnist': Config(
        epochs=6,
        folds=5,
        assessor_test_set_size=10000,
        model=shared.baselines.mnist.model,
        assessor=experiments.kfold.assessors.mnist
    ),
    'cifar10': Config(
        epochs=10,
        folds=5,
        assessor_test_set_size=10000,
        model=shared.baselines.cifar10.model_with(depth=16, width_multiplier=1),
        assessor=experiments.kfold.assessors.mnist
    )
}


@dataclass
class Flags:
    retrain: bool = False
    skip_assessor: bool = False
    remake_dataset: bool = False
    dataset: str = 'mnist'

    def validate(self):
        if self.dataset not in ['mnist', 'cifar10']:
            return ValueError(f'Unknown dataset {self.dataset}')

    def __post_init__(self):
        self.validate()


@click.command()
@click.option('-d', '--dataset', default='mnist', help='Which dataset to use for the experiment.')
@click.option('-r', '--retrain', is_flag=True, help='Wether the models should be retrained even if they can be reloaded from checkpoints (default False)')
@click.option('-s', '--skip-assessor', is_flag=True, help='Wether traingin the assessor model should be skipped (default False)')
@click.option('--remake-dataset', '--rd', is_flag=True, help='Whether the assessor dataset should be recreated even if it can loaded from disk (default False)')
def main(**kwargs):
    flags = Flags(**kwargs)
    config = CONFIG[flags.dataset]

    ds, ds_info = load_dataset(flags)
    ds: Dataset = ds_extra.enumerate_dict(ds)
    folds = ds_extra.k_folds(ds, 5)
    models = train_models(folds, flags)

    if flags.skip_assessor:
        return

    # wandb.init(dir=path.dirname("assets/wandb"), project="assessor-kfold")
    # wandb_callback = WandbCallback(log_evaluation=True)

    ds_path = f"assets/data/kfold/{flags.dataset}"
    if path.exists(ds_path) and not flags.remake_dataset:
        print(f"Loading existing assessor dataset at {ds_path}")
        ass_ds = tf.data.experimental.load(ds_path)
    else:
        ass_ds = create_assessor_dataset(folds, models, ds_info)
        print(f"Saving assessor dataset to {ds_path}")
        tf.data.experimental.save(ass_ds, ds_path)
        print(f"Saved assessor dataset to {ds_path}")

    (ass_test_ds, ass_train_ds) = ds_extra.split_absolute(ass_ds, config.assessor_test_set_size)
    assessor = config.assessor()
    features = {'x': 'image', 'y': 'loss'}
    assessor.fit(
        train_pipeline(ds_extra.to_supervised(ass_train_ds, **features)),
        epochs=config.epochs,
        validation_data=test_pipeline(ds_extra.to_supervised(ass_test_ds, **features)),
        # callbacks=[WandbCallback(predictions=10, generator=ass_test_ds, input_type='image')]
        # callbacks=[wandb_callback]
    )

    # wandb.finish()


def create_assessor_dataset(folds, models: List[Model], ds_info: DatasetInfo) -> tf.data.Dataset:
    """
    Create the assessor dataset, which is the concatenation off all K test sets
    where the label is the loss for the corresponding model
    """
    results: List[Dataset] = []

    for ((_train, test), model) in zip(folds, models):
        # This function maps the entries in the test sets of each fold to
        # a dataset entry fit for training the assessor.
        # The key datapoints are x (i.e. image) and loss which will be the label
        # for the assessor.
        def to_assessor_entry(entry):
            x, y_true = entry['image'], entry['label']
            y_pred = model(x.reshape((1) + x.shape))
            loss = model.loss(y_true, y_pred)
            return entry | {'prediction': y_pred, 'loss': loss}

        results.append(test.map(to_assessor_entry))

    ass_ds: tf.data.Dataset = ds_extra.concatenate_all(results)
    assert ass_ds.cardinality() == ds_info.splits.total_num_examples
    return ass_ds


def load_dataset(flags: Flags) -> Tuple[Dataset, DatasetInfo]:
    # https://www.tensorflow.org/datasets/api_docs/python/tfds/load

    ds_info: DatasetInfo
    (ds_train, ds_test), ds_info = tfds.load(
        flags.dataset,
        with_info=True,

        # We need to concat later, but if we don't split,
        # we get a {"train": Dataset, "test": Dataset} object anyway.
        split=["train", "test"],

        # We explicitly want control over shuffling ourselves
        shuffle_files=False,

        read_config=ReadConfig(
            # We don't want to cache at this stage, since we will build an
            # dataset pipeline on it first.
            try_autocache=False,
        )
    )

    ds: Dataset = ds_train.concatenate(ds_test)
    return (ds, ds_info)


def train_models(folds, flags: Flags) -> List[Model]:
    models: List[Model] = []
    for (i, (train, test)) in enumerate(folds):
        config = CONFIG[flags.dataset]
        model = config.model()

        checkpoint_dir = f"assets/models/kfold/{flags.dataset}/{i}/"
        checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        checkpoint_callback = cb_extra.EpochCheckpointManagerCallback(manager)

        latest = manager.latest_checkpoint
        initial_epoch = 0
        if latest and not flags.retrain:
            print("Using model checkpoint {}".format(latest))

            # We add .expect_partial(), because if a previous run completed,
            # and we consequently restore from the last checkpoint, no further
            # training is need, and we don't expect to use all variables.
            checkpoint.restore(latest).expect_partial()
            initial_epoch = int(checkpoint.save_counter)
        else:
            if flags.retrain:
                print(f"Training on fold {i} from scratch due to --retrain")
            else:
                print(f"Training on fold {i} from scratch")

        features = {'x': 'image', 'y': 'label'}
        model.fit(
            train_pipeline(ds_extra.to_supervised(train, **features)),
            epochs=config.epochs,
            validation_data=test_pipeline(ds_extra.to_supervised(test, **features)),
            callbacks=[checkpoint_callback],
            initial_epoch=initial_epoch
        )
        models.append(model)
    return models


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
