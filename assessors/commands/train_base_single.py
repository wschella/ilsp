from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.core import ModelDefinition, Restore, Dataset, DatasetDescription
from assessors.hubs import DatasetHub, SystemHub
from assessors.application import cli
from assessors.commands.train_ import TrainArgs


@dataclass
class TrainBaseArgs(TrainArgs):
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        super().validate()
        self.validate_option('dataset', DatasetHub.options())
        self.validate_option('model', SystemHub.options_for(self.dataset))


@cli.command(name='train-base-single')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model variant to train")
@click.option("-r", "--restore", default="full", show_default=True, help="Wether to restore the model if possible. Options [full, checkpoint, off]")
@click.pass_context
def train_base_single(ctx, **kwargs):
    args = TrainBaseArgs(parent=ctx.obj, **kwargs).validated()

    dsds: DatasetDescription = DatasetHub.get(args.dataset)
    dataset: Dataset = dsds.load_all()

    path = Path(f"artifacts/models/{args.dataset}/{args.model}/base/")
    model_def: ModelDefinition = SystemHub.get(args.dataset, args.model)()

    (train, test) = dataset.split_relative(-0.2)
    print(f'Train size: {len(train)}, test size: {len(test)}')
    model = model_def.train(train, validation=test, restore=Restore(path, args.restore))
    model.save(path)
