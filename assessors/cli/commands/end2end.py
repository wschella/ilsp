from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.core import Restore
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
    restore: Restore.Options = "off"
    save: bool = True
    assessor_model: str = "mnist_default"
    output_path: Path = Path("results.csv")
    folds: int = 5

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('base_model', ["default"])
        self.validate_option('assessor_model', ["mnist_default", "mnist_prob", "cifar10_default"])


@cli.command()
@click.argument('dataset')
@click.option('-b', '--base-model', default='default', help="The base model variant to train")
@click.option('-a', '--assessor-model', default='mnist_default', help="The assessor model variant to train")
@click.option('-f', '--folds', default=5, help="The number of folds to use for cross-validation")
@click.option('-o', '--output-path', default=Path('results.csv'), type=click.Path(), help="The file to write the results to")
@click.option('-r', '--restore', default='off', help="Whether to restore models. Options [full, checkpoint, off]")
@click.option('--save/--no-save', default=True, help="Whether to save models")
@click.pass_context
def end2end(ctx, **kwargs):
    """
    Train the baseline model for DATASET.
    Options are:
        - mnist
        - cifar10
    """
    args = End2EndArgs(parent=ctx.obj, **kwargs).validated()

    # Download and prepare relevant dataset
    ctx.invoke(dataset_download, name=args.dataset)

    # Train base population
    print("# Training base population")
    ctx.invoke(
        train_kfold,
        dataset=args.dataset,
        model=args.base_model,
        folds=args.folds,
        restore=args.restore,
        save=args.save
    )

    # Make assessor dataset
    print("# Making assessor dataset")
    ctx.invoke(
        dataset_make,
        dataset=args.dataset,
        model=args.base_model,
        folds=args.folds)
    dataset_path = dataset_make.artifact_location(args.dataset, args.base_model, args.folds)

    # Train assessor
    print("# Training assessor")
    ctx.invoke(
        train_assessor,
        dataset=dataset_path,
        model=args.assessor_model,
        restore=args.restore,
        save=args.save)

    # Evaluate assessor
    print("# Evaluating assessor")
    print(args.output_path)
    ctx.invoke(
        evaluate_assessor,
        dataset=dataset_path,
        model=args.assessor_model,
        output_path=args.output_path,
        overwrite=True)
