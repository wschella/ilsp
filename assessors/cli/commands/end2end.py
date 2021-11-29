from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.core import Restore
from assessors.cli.shared import CommandArguments, DatasetHub, AssessorHub, SystemHub
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
    identifier: str = "k5_r1"
    save: bool = True
    download: bool = True
    overwrite_results: bool = False
    assessor_model: str = "default"
    output_path: Path = Path("results.csv")
    folds: int = 5
    repeats: int = 1

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', DatasetHub.options())
        self.validate_option('base_model', SystemHub.options_for(self.dataset))
        self.validate_option('assessor_model', AssessorHub.options_for(self.dataset))


@cli.command()
@click.argument('dataset')
@click.option('-b', '--base-model', default='default', help="The base model variant to train")
@click.option('-a', '--assessor-model', default='default', required=True, help="The assessor model variant to train")
@click.option('-f', '--folds', default=5, help="The number of folds to use for cross-validation")
@click.option('-r', '--repeats', default=1, help="The number of models to train per fold")
@click.option('-i', '--identifier', required=True, help="The identifier for the assessor")
@click.option('-o', '--output-path', default=Path('results.csv'), type=click.Path(path_type=Path), help="The file to write the results to")
@click.option('--restore-systems', default='off', help="Whether to restore base systems. Options [full, checkpoint, off]")
@click.option('--restore-assessor', default='off', help="Whether to restore assessor model. Options [full, checkpoint, off]")
@click.option('--save/--no-save', default=True, help="Whether to save models")
@click.option('--overwrite-results/--no-overwrite=results', default=False, help="Whether to overwrite the results they exist on the same path")
@click.option('--download/--no-download', default=True, help="Whether to download datasets")
@click.pass_context
def end2end(ctx, **kwargs):
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
        repeats=args.repeats,
        restore=args.restore_systems,
        save=args.save)

    # Make assessor dataset
    print("# Making assessor dataset")
    ctx.invoke(
        dataset_make,
        dataset=args.dataset,
        model=args.base_model,
        folds=args.folds,
        repeats=args.repeats)
    dataset_path = dataset_make.artifact_location(  # type: ignore
        args.dataset, args.base_model, args.folds, args.repeats)

    # Train assessor
    print("# Training assessor")
    ctx.invoke(
        train_assessor,
        dataset=dataset_path,
        dataset_name=args.dataset,
        model=args.assessor_model,
        restore=args.restore_assessor,
        identifier=args.identifier,
        overwrite_results=args.overwrite_results,
        save=args.save,

        # Evaluate assessor
        evaluate=True)

    # # Evaluate assessor
    # print("# Evaluating assessor")
    # print(args.output_path)
    # ctx.invoke(
    #     evaluate_assessor,
    #     dataset=dataset_path,
    #     model=args.assessor_model,
    #     identifier=args.identifier,
    #     output_path=args.output_path,
    #     overwrite=True)
