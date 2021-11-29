from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.utils import dataset_extra as dse
from assessors.core import ModelDefinition, Restore, Dataset, PredictionRecord, DatasetDescription, CustomDatasetDescription
from assessors.cli.shared import CommandArguments, DatasetHub, AssessorHub, SystemHub
from assessors.cli.cli import cli, CLIArgs
from assessors.cli.commands.evaluate import evaluate_assessor


@dataclass
class TrainArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    restore: Restore.Options = "off"

    def validate(self):
        self.parent.validate()
        self.validate_option("restore", ["full", "checkpoint", "off"])

# -----------------------------------------------------------------------------


@dataclass
class TrainBaseArgs(TrainArgs):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        super().validate()
        self.validate_option('dataset', DatasetHub.options())
        self.validate_option('model', SystemHub.options_for(self.dataset))


@cli.command(name='train-base')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.pass_context
def train_base(ctx, **kwargs):
    args = TrainBaseArgs(parent=ctx.obj, **kwargs).validated()

    dsds: DatasetDescription = DatasetHub.get(args.dataset)
    dataset: Dataset = dsds.load_all()

    path = Path(f"artifacts/models/{args.dataset}/{args.model}/base/")
    model_def: ModelDefinition = SystemHub.get(args.dataset, args.model)()

    (train, test) = dataset.split_relative(-0.2)
    print(f'Train size: {len(train)}, test size: {len(test)}')
    model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
    model.save(path)


# -----------------------------------------------------------------------------


@dataclass
class TrainKFoldArgs(TrainArgs):
    folds: int = 5
    repeats: int = 1
    dataset: str = "mnist"
    model: str = "default"
    save: bool = True

    def validate(self):
        super().validate()
        self.validate_option('dataset', DatasetHub.options())
        self.validate_option('model', SystemHub.options_for(self.dataset))


@cli.command(name='train-kfold')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="The number of folds")
@click.option('-r', '--repeats', default=1, help="The number of models that will be trained for each fold")
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.option("--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.option("--save/--no-save", default=True, show_default=True, help="Wether to save the model")
@click.pass_context
def train_kfold(ctx, **kwargs):
    """
    Train the baseline model for DATASET. Options are: [mnist, cifar10].
    """
    args = TrainKFoldArgs(parent=ctx.obj, **kwargs).validated()

    model_def: ModelDefinition = SystemHub.get(args.dataset, args.model)()

    dsds: DatasetDescription = DatasetHub.get(args.dataset)
    dataset: Dataset = dsds.load_all()

    base_path = Path(
        f"artifacts/systems/{args.dataset}/{args.model}/kfold_f{args.folds}_r{args.repeats}/")
    for i, (train, test) in enumerate(dse.k_folds(dataset, args.folds)):
        for repeat in range(args.repeats):
            print(f'Fold {i+1}/{args.folds}, repeat {repeat+1}/{args.repeats}')
            path = base_path / f"fold_{i}" / f"model_{repeat}"
            model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
            if args.save:
                model.save(path)


# -----------------------------------------------------------------------------

@dataclass
class TrainAssessorArgs(TrainArgs):
    dataset: Path = Path("artifacts/datasets/mnist/kfold/")
    dataset_name: str = "mnist"
    model: str = "default"
    identifier: str = "k5_r1"
    save: bool = True
    evaluate: bool = True
    overwrite_results: bool = False

    def validate(self):
        super().validate()
        self.validate_option('dataset_name', DatasetHub.options())
        self.validate_option('model', AssessorHub.options_for(self.dataset_name))


@cli.command(name='train-assessor')
@click.argument('dataset', type=click.Path(exists=True, path_type=Path))
@click.option('-d', '--dataset-name', required=True, help="The name of the source dataset")
@click.option('-m', '--model', required=True, help="The model variant to train.")
@click.option('-i', '--identifier', required=True, help="The identifier of assessor (for saving path).")
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the assessor if possible. Options [full, checkpoint, off]")
@click.option("--save/--no-save", default=True, show_default=True, help="Wether to save the assessor")
@click.option("--evaluate/--no-evaluate", default=True, show_default=True, help="Wether to evaluate the model")
@click.option("--overwrite-results/--no-overwrite-results", default=False, show_default=True, help="Wether to overwrite the results")
@click.pass_context
def train_assessor(ctx, **kwargs):
    """
    Train the assessor model for dataset at DATASET.
    """
    args = TrainAssessorArgs(parent=ctx.obj, **kwargs).validated()
    model_def: ModelDefinition = AssessorHub.get(args.dataset_name, args.model)()

    def to_supervised(record: PredictionRecord):
        return (record['inst_features'], record['syst_pred_score'])

    _dataset: Dataset[PredictionRecord, Any] = CustomDatasetDescription(
        path=args.dataset).load_all()
    supervised = _dataset.map(to_supervised)

    path = Path(f"artifacts/assessors/{args.dataset_name}/{args.model}/{args.identifier}/")

    (train, test) = supervised.split_relative(-0.2)
    print(f'Train size: {len(train)}, test size: {len(test)}')
    print(f'Training {model_def.name()}')
    model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
    if args.save:
        model.save(path)

    ctx.invoke(
        evaluate_assessor,
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        model=args.model,
        identifier=args.identifier,
        overwrite=args.overwrite_results)
