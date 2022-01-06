from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.utils import dataset_extra as dse
from assessors.core import ModelDefinition, Restore, Dataset, DatasetDescription
from assessors.hubs import DatasetHub, SystemHub
from assessors.application import cli
from assessors.commands.train_ import TrainArgs


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


@cli.command(name='train-base-kfold')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="The number of folds")
@click.option('-r', '--repeats', default=1, help="The number of models that will be trained for each fold")
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.option("--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.option("--save/--no-save", default=True, show_default=True, help="Wether to save the model")
@click.pass_context
def train_base_kfold(ctx, **kwargs):
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
