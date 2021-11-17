from typing import *
from dataclasses import dataclass
from pathlib import *

import click

from assessors.core import ModelDefinition, PredictionRecord
from assessors.core import Dataset, DatasetDescription
from assessors.utils import dataset_extra as dse
from assessors.cli.shared import CommandArguments, get_dataset_description, get_model_def
from assessors.cli.cli import cli, CLIArgs


@dataclass
class DownloadArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    name: str = "mnist"

    def validate(self):
        self.parent.validate()


@cli.command('dataset-download')
@click.argument('name')
@click.pass_context
def dataset_download(ctx, **kwargs):
    """
    Download dataset NAME from tensorflow datasets. Happens automatically as required as well.
    See https://www.tensorflow.org/datasets/catalog/overview for an overview of options.
    """
    args = DownloadArgs(parent=ctx.obj, **kwargs).validated()
    dataset: DatasetDescription = get_dataset_description(args.name)
    dataset.download()


@dataclass
class MakeKFoldArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"
    folds: int = 5
    repeats: int = 1

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10", "segment"])
        self.validate_option('model', ["default"])


@cli.command('dataset-make')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="Number of folds to use.")
@click.option('-r', '--repeats', default=1, help="Number of models per fold")
@click.option('-m', '--model', default="default", help="The model variant to use.")
@click.pass_context
def dataset_make(ctx, **kwargs):
    """
    Makes an assessor dataset from a KFold-trained collection of models.
    """
    args = MakeKFoldArgs(parent=ctx.obj, **kwargs).validated()

    model_def: ModelDefinition = get_model_def(args.dataset, args.model)()
    dataset_desc: DatasetDescription = get_dataset_description(args.dataset)

    dataset: Dataset = dataset_desc.load_all()
    dataset = dataset.map(lambda e: {'features': e[0], 'target': e[1]}).enumerate_dict()

    models = []
    ds_parts = []
    dir = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold_f{args.folds}_r{args.repeats}/")
    n_folds = len(list(dir.glob("*")))

    # TODO: Fix non batched inference
    for i, (_train, test) in enumerate(dse.k_folds(dataset, n_folds)):
        for repeat in range(args.repeats):
            path = dir / f"fold_{i}" / f"model_{repeat}"
            model = model_def.restore_from(path)
            if model is None:
                raise ValueError(f"No model found at {path}")

            # We need to keep a reference to the model because otherwise TF
            # prematurely deletes it.
            # https://github.com/OpenNMT/OpenNMT-tf/pull/842
            models.append(model)

            def to_prediction_record(entry) -> PredictionRecord:
                x, y_true = entry['features'], entry['target']
                y_pred = model(x)
                return PredictionRecord(
                    inst_index=entry['index'],
                    inst_features=entry['features'],
                    inst_target=entry['target'],
                    syst_features=test.encode([i]),
                    syst_prediction=y_pred,
                    syst_pred_loss=model.loss(y_true, y_pred),
                    syst_pred_score=model.score(y_true, y_pred),
                )

            part = test.map(to_prediction_record)
            ds_parts.append(part)

    print("Saving assessor model dataset. This is currently quite slow because we're doing non batched inference")
    assessor_ds: Dataset = ds_parts[0].interleave_with(ds_parts[1:], cycle_length=n_folds)
    assessor_ds_path = dataset_make.artifact_location(   # type: ignore
        args.dataset, args.model, n_folds, args.repeats)
    assessor_ds.save(assessor_ds_path)


# Add an attribute to the function / command that tells where it will store the artifact
cast(Any, dataset_make).artifact_location = \
    lambda dataset, model, n_folds, n_repeats: \
    Path(f"artifacts/datasets/{dataset}/{model}/kfold_f{n_folds}_r{n_repeats}/")
