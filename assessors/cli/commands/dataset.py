from assessors.models.model import BaseModel
from typing import *
from dataclasses import dataclass
from pathlib import *

import click
from tqdm import tqdm

import tensorflow as tf

from assessors.datasets import TFDatasetWrapper
from assessors.utils import dataset_extra as dse
from assessors.cli.shared import CommandArguments, get_model
from assessors.cli.cli import cli, CLIArgs


@dataclass
class DownloadArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    name: str = "mnist"

    def validate(self):
        self.parent.validate()


@cli.command('dataset-download')
@click.argument('name')
@click.pass_context
def dataset_download(ctx, **kwargs):
    """
    Download dataset NAME from tensorflow datasets. Happens automatically as required as well.
    See https://www.tensorflow.org/datasets/catalog/overview for an overview of options.
    """
    args = DownloadArgs(parent=ctx.obj, **kwargs).validated()
    dataset = TFDatasetWrapper(args.name)
    dataset.load()


@dataclass
class MakeKFoldArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"
    folds: int = 5

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@cli.command('dataset-make')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="The number of folds.")
@click.option('-m', '--model', default="default", help="The model variant to train.")
@click.pass_context
def dataset_make(ctx, **kwargs):
    """
    Makes an assessor dataset from a KFold-trained collection of models.
    """
    args = MakeKFoldArgs(parent=ctx.obj, **kwargs).validated()

    model_class: type[BaseModel] = get_model(args.dataset, args.model)

    dataset = TFDatasetWrapper(args.dataset, as_supervised=False).load_all()
    dataset = dse.enumerate_dict(dataset)

    models = []
    assessor_dataset_parts = []

    for i, (_train, test) in enumerate(dse.k_folds(dataset, args.folds)):
        path = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold/{i}")
        model: BaseModel = model_class(path=path, restore="full")
        model.load()

        # We need to keep a reference to the model because otherwise TF
        # prematurely deletes it.
        # https://github.com/OpenNMT/OpenNMT-tf/pull/842
        models.append(model)

        def to_assessor_entry(entry):
            x, y_true = normalize_img(entry['image'], entry['label'])
            y_pred = model(x.reshape((1) + x.shape))
            loss = model.loss(y_true, y_pred)
            entry = entry | {'prediction': y_pred, 'loss': loss}
            return entry

        part = test.map(to_assessor_entry)
        assessor_dataset_parts.append(part)

    assessor_dataset = dse.concatenate_all(assessor_dataset_parts)
    assert assessor_dataset.cardinality() == dataset.cardinality()

    assessor_dataset_path = Path(
        f"artifacts/datasets/{args.dataset}/{args.model}/kfold_{args.folds}/")
    print("Saving assessor model dataset. This is super slow because TF dataset API sucks a bit.")
    tf.data.experimental.save(assessor_dataset, str(assessor_dataset_path))


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
