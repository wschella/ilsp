from typing import *
from dataclasses import dataclass
from pathlib import *

import click

import tensorflow as tf

from assessors.core import ModelDefinition, TFDatasetWrapper, PredictionRecord
from assessors.utils import dataset_extra as dse
from assessors.cli.shared import CommandArguments, get_model_def
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
    dataset = TFDatasetWrapper(args.name)
    dataset.load()


@dataclass
class MakeKFoldArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    model: str = "default"
    folds: int = 5

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset', ["mnist", "cifar10"])
        self.validate_option('model', ["default"])


@cli.command('dataset-make')
@click.argument('dataset')
@click.option('-f', '--folds', default=5, help="Number of folds to use.")
@click.option('-m', '--model', default="default", help="The model variant to use.")
@click.pass_context
def dataset_make(ctx, **kwargs):
    """
    Makes an assessor dataset from a KFold-trained collection of models.
    """
    args = MakeKFoldArgs(parent=ctx.obj, **kwargs).validated()

    model_def: ModelDefinition = get_model_def(args.dataset, args.model)()

    dataset = TFDatasetWrapper(args.dataset, as_supervised=False).load_all()
    # TODO: This is not working, it enumerates last.
    dataset = dse.enumerate_dict(dataset)

    models = []
    ds_parts = []
    dir = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold_{args.folds}/")
    n_folds = len(list(dir.glob("*")))

    # TODO: Fix non batched inference
    for i, (_train, test) in enumerate(dse.k_folds(dataset, n_folds)):
        path = dir / str(i)
        model = model_def.try_restore_from(path)

        # We need to keep a reference to the model because otherwise TF
        # prematurely deletes it.
        # https://github.com/OpenNMT/OpenNMT-tf/pull/842
        models.append(model)

        def to_prediction_record(entry) -> PredictionRecord:
            x, y_true = normalize_img(entry['image'], entry['label'])
            y_pred = model(x.reshape((1) + x.shape))
            return {
                'inst_index': entry['index'],
                'inst_features': entry['image'],
                'inst_label': entry['label'],
                'syst_features': i,
                'syst_prediction': y_pred,
                'syst_pred_loss': model.loss(y_true, y_pred),
                'syst_pred_score': model.score(y_true, y_pred),
            }

        part = test.map(to_prediction_record)
        ds_parts.append(part)

    assessor_ds = dse.concatenate_all(ds_parts)
    assert assessor_ds.cardinality() == dataset.cardinality()

    print("Saving assessor model dataset. This is currently super slow because we're doing non batched inference")
    assessor_ds_path = dataset_make.artifact_location(args.dataset, args.model, n_folds)
    tf.data.experimental.save(assessor_ds, str(assessor_ds_path))


# Add an attribute to the function / command that tells where it will store the artifact
cast(Any, dataset_make).artifact_location = lambda dataset, model, n_folds: Path(
    f"artifacts/datasets/{dataset}/{model}/kfold_{n_folds}/")


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
