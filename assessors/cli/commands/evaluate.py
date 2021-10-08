from typing import *
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import click

from assessors.datasets import TFDatasetWrapper

from assessors.cli.shared import CommandArguments, get_model
from assessors.cli.cli import cli, CLIArgs


@dataclass
class EvaluateArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()

    def validate(self):
        self.parent.validate()


@cli.group()
@click.pass_context
def evaluate(ctx, **kwargs):
    """
    Evaluate a model on a dataset.
    """
    ctx.obj = EvaluateArgs(parent=ctx.obj, **kwargs).validated()


@dataclass
class EvaluateBaseArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@evaluate.command(name='base')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.pass_context
def evaluate_base(ctx, **kwargs):
    args = EvaluateBaseArgs(parent=ctx.obj, **kwargs).validated()
    dataset = TFDatasetWrapper(args.dataset)
    (_train, test) = itemgetter('train', 'test')(dataset.load())

    path = Path(f"artifacts/models/mnist/base")
    model_class = get_model(args.dataset, args.model)
    model = model_class(path=path, restore="full")

    model.load()
    model.evaluate(test)
