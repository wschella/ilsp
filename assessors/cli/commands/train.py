from assessors.models.model import BaseModel, Restore
from typing import *
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import click

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.datasets import TFDatasetWrapper, CustomDataset
from assessors.utils import dataset_extra as dse
from assessors.cli.shared import CommandArguments, get_model, get_assessor
from assessors.cli.cli import cli, CLIArgs


@dataclass
class TrainArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    restore: str = "full"

    def validate(self):
        self.parent.validate()
        self.validate_option("restore", ["full", "checkpoint", "off"])


@cli.group()
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.pass_context
def train(ctx, **kwargs):
    """
    Train a model on a dataset. You can train base models and assessor models.
    """
    ctx.obj = TrainArgs(parent=ctx.obj, **kwargs).validated()

# -----------------------------------------------------------------------------


@dataclass
class TrainBaseArgs(CommandArguments):
    parent: TrainArgs = TrainArgs()
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@train.command(name='base')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.pass_context
def train_base(ctx, **kwargs):
    """
    Train the baseline model for DATASET.
    Options are:
        - mnist
        - cifar10
    """
    args = TrainBaseArgs(parent=ctx.obj, **kwargs).validated()

    dataset = TFDatasetWrapper(args.dataset)

    path = Path(f"artifacts/models/{args.dataset}/{args.model}/base/")
    model_class = get_model(args.dataset, args.model)
    model = model_class(path=path, restore=args.parent.restore)

    (train, test) = itemgetter('train', 'test')(dataset.load())
    model.train(train, validation=test)


# -----------------------------------------------------------------------------


@dataclass
class TrainKFoldArgs(CommandArguments):
    parent: TrainArgs = TrainArgs()
    folds: int = 5
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@train.command(name='kfold')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="The number of folds")
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.pass_context
def train_kfold(ctx, **kwargs):
    """
    Train the baseline model for DATASET. Options are: [mnist, cifar10].
    """
    args = TrainKFoldArgs(parent=ctx.obj, **kwargs).validated()

    model_class = get_model(args.dataset, args.model)

    dataset = TFDatasetWrapper(args.dataset).load_all()
    for i, (train, test) in enumerate(dse.k_folds(dataset, args.folds)):
        path = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold/{i}")
        model = model_class(path=path, restore=args.parent.restore)
        model.train(train, validation=test)


# -----------------------------------------------------------------------------

@dataclass
class TrainAssessorArgs(CommandArguments):
    parent: TrainArgs = TrainArgs()
    dataset: Path = Path("artifacts/datasets/mnist/kfold/")
    test_size: int = 10000
    model: str = "mnist_default"

    def validate(self):
        self.parent.validate()
        self.validate_option('model', ["mnist_default", "mnist_prob"])


@train.command(name='assessor')
@click.argument('dataset', type=click.Path(exists=True))
@click.option('-s', '--test-size', default=10000, help="The size of the test split.")
@click.option('-m', '--model', default='mnist_default', help="The model variant to train.")
@click.pass_context
def train_assessor(ctx, **kwargs):
    """
    Train the assessor model for dataset at DATASET.
    """
    args = TrainAssessorArgs(parent=ctx.obj, **kwargs).validated()

    [dataset_name, model_name] = args.model.split('_')
    model_class: type[BaseModel] = get_assessor(dataset_name, model_name)

    def to_binary_result(entry):
        (pred, y_true) = entry["prediction"], entry["label"]
        return entry | {"bin_result": tf.math.argmax(pred, axis=1) == y_true}

    dataset: TFDataset = CustomDataset(path=args.dataset).load_all()
    dataset = dse.to_supervised(dataset.map(to_binary_result), x="image", y="bin_result")

    path = Path(f"artifacts/models/{dataset}/{args.model}/assessor/")
    model: BaseModel = model_class(path=path, restore=args.parent.restore)

    (train, test) = dse.split_absolute(dataset, dataset.cardinality() - args.test_size)
    model.train(train, validation=test)
