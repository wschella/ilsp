from typing import *
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import sklearn.metrics as metrics

from assessors.core import ModelDefinition, Dataset
from assessors.utils.cli import CommandArguments
from assessors.hubs import SystemHub, DatasetHub
from assessors.application import cli, CLIArgs


@dataclass
class EvaluateSystemArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', DatasetHub.options())
        self.validate_option('model', SystemHub.options_for(self.dataset))


@cli.command(name='eval-system')
@click.argument('dataset')
@click.option('-m', '--model', default='default', help="The model to evaluate")
@click.pass_context
def evaluate_system(ctx, **kwargs):
    # Handle CLI args
    args = EvaluateSystemArgs(parent=ctx.obj, **kwargs).validated()

    # Load system
    model_def: ModelDefinition = SystemHub.get(args.dataset, args.model)()
    # TODO: Fix
    model_path = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold_5/4")
    model = model_def.restore_from(model_path)

    # Load & mangle dataset
    _ds: Dataset = DatasetHub.get(args.dataset).load_all()
    (_train, test) = _ds.split_relative(-0.2)

    y_true = test.map(lambda e: e[1]).as_numpy()
    y_pred = model.predict_all(test.map(lambda e: e[0]))
    y_pred = np.argmax(y_pred, axis=1)

    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))
