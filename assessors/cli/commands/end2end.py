from typing import *
from dataclasses import dataclass

import click


from assessors.cli.shared import CommandArguments
from assessors.cli.cli import cli, CLIArgs

from assessors.cli.commands.dataset import dataset_download, dataset_make
from assessors.cli.commands.train import train_kfold, train_assessor
from assessors.cli.commands.evaluate import evaluate_assessor


@dataclass
class End2EndArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    base_model: str = "default"
    assessor_model: str = "mnist_default"
    folds: int = 5

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('base_model', ["default"])
        self.validate_option('assessor_model', ["mnist_default", "mnist_prob"])


@cli.command()
@click.argument('dataset')
@click.option('-b', '--base-model', default='default', help="The base model variant to train")
@click.option('-a', '--assessor-model', default='mnist_default', help="The assessor model variant to train")
@click.pass_context
def end2end(ctx, **kwargs):
    """
    Train the baseline model for DATASET.
    Options are:
        - mnist
        - cifar10
    """
    args = End2EndArgs(parent=ctx.obj, **kwargs).validated()

    ctx.invoke(dataset_download, name=args.dataset)
    ctx.invoke(train_kfold, dataset=args.dataset,
               model=args.base_model, folds=args.folds, restore="off")
    ctx.invoke(dataset_make, dataset=args.dataset, model=args.base_model)
    dataset_path = dataset_make.artifact_location(args.dataset, args.base_model, args.folds)

    ctx.invoke(train_assessor, dataset=dataset_path, model=args.assessor_model, restore="off")
    ctx.invoke(evaluate_assessor, dataset=dataset_path, model=args.assessor_model)
