from typing import *
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import click

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.datasets import TFDatasetWrapper, CustomDataset
from assessors.utils import dataset_extra as dse
from assessors.models import ModelDefinition, TrainedModel, Restore
from assessors.cli.shared import CommandArguments, get_model_def, get_assessor_def
from assessors.cli.cli import cli, CLIArgs


@dataclass
class TrainArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    restore: str = "full"

    def validate(self):
        self.parent.validate()
        self.validate_option("restore", ["full", "checkpoint", "off"])

# -----------------------------------------------------------------------------


@dataclass
class TrainBaseArgs(TrainArgs):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"
    test_size: int = 10000

    def validate(self):
        super().validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@cli.command(name='train-base')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.option('-t', '--test-size', default=10000, help="The size of the test set")
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.pass_context
def train_base(ctx, **kwargs):
    """
    Train the baseline model for DATASET.
    Options are:
        - mnist
        - cifar10
    """
    args = TrainBaseArgs(parent=ctx.obj, **kwargs).validated()

    dataset = TFDatasetWrapper(args.dataset).load_all()

    path = Path(f"artifacts/models/{args.dataset}/{args.model}/base/")
    model_def: ModelDefinition = get_model_def(args.dataset, args.model)()

    (train, test) = dse.split_absolute(dataset, dataset.cardinality() - args.test_size)
    print(f'Train size: {len(train)}, test size: {len(test)}')
    model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
    model.save(path)


# -----------------------------------------------------------------------------


@dataclass
class TrainKFoldArgs(TrainArgs):
    folds: int = 5
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        super().validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@cli.command(name='train-kfold')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="The number of folds")
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.pass_context
def train_kfold(ctx, **kwargs):
    """
    Train the baseline model for DATASET. Options are: [mnist, cifar10].
    """
    args = TrainKFoldArgs(parent=ctx.obj, **kwargs).validated()

    model_def: ModelDefinition = get_model_def(args.dataset, args.model)()

    dataset = TFDatasetWrapper(args.dataset).load_all()
    for i, (train, test) in enumerate(dse.k_folds(dataset, args.folds)):
        path = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold_{args.folds}/{i}")
        model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
        model.save(path)


# -----------------------------------------------------------------------------

@dataclass
class TrainAssessorArgs(TrainArgs):
    dataset: Path = Path("artifacts/datasets/mnist/kfold/")
    test_size: int = 10000
    model: str = "mnist_default"

    def validate(self):
        super().validate()
        self.validate_option('model', ["mnist_default", "mnist_prob"])


@cli.command(name='train-assessor')
@click.argument('dataset', type=click.Path(exists=True))
@click.option('-s', '--test-size', default=10000, help="The size of the test split.")
@click.option('-m', '--model', default='mnist_default', help="The model variant to train.")
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.pass_context
def train_assessor(ctx, **kwargs):
    """
    Train the assessor model for dataset at DATASET.
    """
    args = TrainAssessorArgs(parent=ctx.obj, **kwargs).validated()

    [dataset_name, model_name] = args.model.split('_')
    model_def: ModelDefinition = get_assessor_def(dataset_name, model_name)()

    def to_binary_result(entry):
        (pred, y_true) = entry["prediction"], entry["label"]
        return entry | {"bin_result": tf.math.argmax(pred, axis=1) == y_true}

    dataset: TFDataset = CustomDataset(path=args.dataset).load_all()
    dataset = dse.to_supervised(dataset.map(to_binary_result), x="image", y="bin_result")

    path = Path(f"artifacts/models/{dataset_name}/{model_name}/assessor/")

    (train, test) = dse.split_absolute(dataset, dataset.cardinality() - args.test_size)
    model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
    model.save(path)
