from typing import Any, List, Tuple, Callable
from dataclasses import dataclass
from os import path
import logging

# We need to do this before importing TF
from shared.util.tf_logging import set_tf_loglevel
set_tf_loglevel(logging.WARN)  # nopep8

import click
import wandb
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras
import tensorflow_datasets as tfds

from tensorflow.keras import Model
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow_datasets.core import DatasetInfo
from tensorflow.python.framework.ops import Tensor
from tensorflow_datasets.core.utils.read_config import ReadConfig
from tensorflow.python.ops.numpy_ops import np_config

from wandb.keras import WandbCallback
from click_option_group import optgroup

import shared.util.callbacks_extra as cb_extra
import shared.util.dataset_extra as ds_extra
import shared.util.click_extra as click_extra
import shared

import experiments.kfold.assessors

# Used for .flatten() on Tensors
np_config.enable_numpy_behavior()  # nopep8

Dataset = tf.data.Dataset

CHECKPOINT_DIR_BASE = lambda dataset: f"assets/models/kfold/{dataset}/"
CHECKPOINT_DIR_DATASET = lambda dataset: f"assets/data/kfold/{dataset}/"
CHECKPOINT_DIR_ASSESSOR = lambda dataset: f"assets/models/kfold/{dataset}_assessor/"


@dataclass
class Config:
    epochs: int
    folds: int
    assessor_test_set_size: int
    assessor_epochs: int
    model: Callable[[], Model]
    assessor: Callable[[], Model]

    def override(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)


CONFIG = {
    'mnist': Config(
        epochs=6,
        folds=5,
        assessor_test_set_size=10000,
        assessor_epochs=15,
        model=shared.baselines.mnist.model,
        assessor=experiments.kfold.assessors.mnist
    ),
    'cifar10': Config(
        epochs=10,
        folds=5,
        assessor_test_set_size=10000,
        assessor_epochs=50,
        model=shared.baselines.cifar10.model_with(depth=16, width_multiplier=1),
        assessor=lambda: experiments.kfold.assessors.cifar10(depth=16, width_multiplier=1)
    )
}


@dataclass
class Flags:
    restore_base: bool = True
    restore_assessor: bool = True
    restore_dataset: bool = True
    skip_assessor: bool = False
    dataset: str = 'mnist'

    @staticmethod
    def collapse(kwargs):
        restore = kwargs['restore']
        del kwargs['restore']

        if restore is None:
            return kwargs

        for asset in ['restore_base', 'restore_assessor', 'restore_dataset']:
            kwargs[asset] = restore
        return kwargs

    def validate(self):
        if self.dataset not in ['mnist', 'cifar10']:
            return ValueError(f'Unknown dataset {self.dataset}')

    def __post_init__(self):
        self.validate()


@click.command()
@click.option('-d', '--dataset', default='mnist', help='Which dataset to use for the experiment.')
@click.option('--restore/--no-restore', default=None, help='Wether everything (see specific flags) should be restored from checkpoints if possible. Can NOT be overridden by specific flags.', show_default=True)
@click.option('--restore-base/--no-restore-base', default=True, help='Wether the base models should be restored from checkpoints if possible', show_default=True)
@click.option('--restore-assessor/--no-restore-assessor', default=True, help='Whether the assessor model should be should be restored from checkpoints if possible', show_default=True)
@click.option('--restore-dataset/--no-restore-dataset', default=True, help='Whether the assessor dataset should be should be restored from checkpoints if possible', show_default=True)
@click.option('-s', '--skip-assessor', is_flag=True, help='Wether training the assessor model should be skipped', show_default=True)
@optgroup.group('Configuration and Hyperparameters')
@click_extra.options_from_dataclass(Config, prefix="c-", with_optgroup=True, exclude=["model", "assessor"])
def main(**kwargs):
    # Load the flags (for and configs
    [flag_kwargs, config_kwargs] = click_extra.split_arguments(kwargs, prefixes=['c_'])
    flags = Flags(**(Flags.collapse(flag_kwargs)))
    config = CONFIG[flags.dataset]
    config.override(**config_kwargs)

    # Train K baseline models
    ds, ds_info = load_dataset(flags)
    ds: Dataset = ds_extra.enumerate_dict(ds)
    folds = ds_extra.k_folds(ds, 5)
    models = train_models(folds, flags)

    if flags.skip_assessor:
        return

    # wandb.init(dir=path.dirname("assets/wandb"), project="assessor-kfold")
    # wandb_callback = WandbCallback(log_evaluation=True)

    # Create the assessor dataset (or restore)
    ds_path = CHECKPOINT_DIR_DATASET(flags.dataset)
    creator = lambda: create_assessor_dataset(folds, models, ds_info)
    ass_ds = restore_or_create_dataset(
        ds_path, flags.restore_dataset, creator, reload_after_save=True)
    ass_ds = ds_extra.to_supervised(ass_ds, x="image", y="loss")
    (ass_test_ds, ass_train_ds) = ds_extra.split_absolute(ass_ds, config.assessor_test_set_size)

    # Create assessor model (or restore)
    assessor: Model = config.assessor()
    checkpoint_dir = CHECKPOINT_DIR_ASSESSOR(flags.dataset)
    assessor, checkpoint_callback, initial_epoch = try_restore_model(
        assessor, checkpoint_dir, flags.restore_assessor)

    assessor.fit(
        train_pipeline(ass_train_ds),
        epochs=config.assessor_epochs,
        validation_data=test_pipeline(ass_test_ds),
        callbacks=[
            checkpoint_callback
        ],
        initial_epoch=initial_epoch
    )

    assessor.evaluate(test_pipeline(ass_test_ds))


def restore_or_create_dataset(ds_path: str, should_restore: bool, creator: Callable, reload_after_save: bool = True) -> tf.data.Dataset:

    if path.exists(ds_path) and should_restore:
        print(f"Loading existing dataset at {ds_path}")
        ds = tf.data.experimental.load(ds_path)

    else:
        ds = creator()
        print(f"Saving dataset to {ds_path}")
        tf.data.experimental.save(ds, ds_path)
        print(f"Saved dataset to {ds_path}")
        ds = tf.data.experimental.load(ds_path)
        print(f"Now loaded assessor dataset from disk")

    return ds


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
            x, y_true = normalize_img(entry['image'], entry['label'])
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

        checkpoint_dir = f"{CHECKPOINT_DIR_BASE(flags.dataset)}/{i}/"
        model, checkpoint_callback, initial_epoch = try_restore_model(
            model, checkpoint_dir, flags.restore_base)

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


def try_restore_model(model: Model, checkpoint_dir: str, restore: bool = True) -> Tuple[Model, Any, int]:
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
    checkpoint_callback = cb_extra.EpochCheckpointManagerCallback(manager)

    latest = manager.latest_checkpoint
    initial_epoch = 0
    if latest and restore:
        print("Using model checkpoint {}".format(latest))

        # We add .expect_partial(), because if a previous run completed,
        # and we consequently restore from the last checkpoint, no further
        # training is need, and we don't expect to use all variables.
        checkpoint.restore(latest).expect_partial()
        initial_epoch = int(checkpoint.save_counter)
    else:
        if not restore:
            print(f"Training on from scratch due to --no-restore for {checkpoint_dir}")
        else:
            print(f"Training on from scratch for {checkpoint_dir}")

    return model, checkpoint_callback, initial_epoch


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
