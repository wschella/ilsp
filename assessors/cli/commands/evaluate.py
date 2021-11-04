from typing import *
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
import csv
import os

import click
import pandas as pd

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.datasets import TFDatasetWrapper, CustomDataset
from assessors.models import ModelDefinition
from assessors.cli.shared import CommandArguments, get_model_def, get_assessor_def
from assessors.cli.cli import cli, CLIArgs
from assessors.utils import dataset_extra as dse


@dataclass
class EvaluateBaseArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@cli.command(name='eval-base')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.pass_context
def evaluate_base(ctx, **kwargs):
    args = EvaluateBaseArgs(parent=ctx.obj, **kwargs).validated()
    dataset = TFDatasetWrapper(args.dataset)
    (_train, test) = itemgetter('train', 'test')(dataset.load())

    model_path = Path(f"artifacts/models/mnist/base")
    model_def: ModelDefinition = get_model_def(args.dataset, args.model)()
    model = model_def.try_restore_from(model_path)

    model.evaluate(test)

# ---------------------------------------------------------------------


@dataclass
class EvaluateAssessorArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: Path = Path("artifacts/datasets/mnist/kfold/")
    test_size: int = 10000
    model: str = "mnist_default"

    def validate(self):
        self.parent.validate()
        self.validate_option('model', ["mnist_default", "mnist_prob"])


@cli.command(name='eval-assessor')
@click.argument('dataset', type=click.Path(exists=True))
@click.option('-m', '--model', default='mnist_default', help="The model to evaluate")
@click.pass_context
def evaluate_assessor(ctx, **kwargs):
    args = EvaluateAssessorArgs(parent=ctx.obj, **kwargs).validated()

    [dataset_name, model_name] = args.model.split('_')
    model_def: ModelDefinition = get_assessor_def(dataset_name, model_name)()
    model_path = Path(f"artifacts/models/{dataset_name}/{model_name}/assessor/")
    model = model_def.try_restore_from(model_path)

    dataset = TFDatasetWrapper(args.dataset)
    dataset: TFDataset = CustomDataset(path=args.dataset).load_all()

    def to_binary_result(entry):
        (pred, y_true) = entry["prediction"], entry["label"]
        return entry | {"bin_result": tf.math.argmax(pred, axis=1) == y_true}

    supervised = dse.to_supervised(dataset.map(to_binary_result), x="image", y="bin_result")
    (_train, test) = dse.split_absolute(supervised, supervised.cardinality() - args.test_size)

    path = Path(f"artifacts/results/{dataset_name}_{model_name}_assessor.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as csvfile:
        model.evaluate(test, csvfile)

    with open(path, 'r', newline='') as csvfile:
        df = pd.read_csv(csvfile)
        print(df.describe())
        print(df.head(10))
