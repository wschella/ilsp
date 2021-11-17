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
    restore_systems: Restore.Options = "off"
    restore_assessor: Restore.Options = "off"
    test_size: int = 10000
    save: bool = True
    download: bool = True
    assessor_model: str = "mnist_default"
    output_path: Path = Path("results.csv")
    folds: int = 5

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10", "segment"])
        self.validate_option('base_model', ["default"])
        self.validate_option('assessor_model', [
                             "mnist_default", "mnist_prob", "cifar10_default", "segment_default"])


@cli.command()
@click.argument('dataset')
@click.option('-b', '--base-model', default='default', help="The base model variant to train")
@click.option('-a', '--assessor-model', default='mnist_default', help="The assessor model variant to train")
@click.option('-f', '--folds', default=5, help="The number of folds to use for cross-validation")
@click.option('-o', '--output-path', default=Path('results.csv'), type=click.Path(path_type=Path), help="The file to write the results to")
@click.option('--restore-systems', default='off', help="Whether to restore base systems. Options [full, checkpoint, off]")
@click.option('--restore-assessor', default='off', help="Whether to restore assessor model. Options [full, checkpoint, off]")
@click.option('-t', '--test-size', default=10000, help="The number of test samples to use for testing the assessor")
@click.option('--save/--no-save', default=True, help="Whether to save models")
@click.option('--download/--no-download', default=True, help="Whether to download datasets")
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
    if args.download:
        ctx.invoke(dataset_download, name=args.dataset)

    # Train base population
    print("# Training base population")
    ctx.invoke(
        train_kfold,
        dataset=args.dataset,
        model=args.base_model,
        folds=args.folds,
        restore=args.restore_systems,
        save=args.save
    )

    # Make assessor dataset
    print("# Making assessor dataset")
    ctx.invoke(
        dataset_make,
        dataset=args.dataset,
        model=args.base_model,
        folds=args.folds)
    dataset_path = dataset_make.artifact_location(  # type: ignore
        args.dataset, args.base_model, args.folds)

    # Train assessor
    print("# Training assessor")
    ctx.invoke(
        train_assessor,
        dataset=dataset_path,
        model=args.assessor_model,
        restore=args.restore_assessor,
        test_size=args.test_size,
        save=args.save)

    # Evaluate assessor
    print("# Evaluating assessor")
    print(args.output_path)
    ctx.invoke(
        evaluate_assessor,
        dataset=dataset_path,
        model=args.assessor_model,
        output_path=args.output_path,
        test_size=args.test_size,
        overwrite=True)
