import logging
from typing import *
from dataclasses import dataclass
from pathlib import Path

import click

from assessors.core import ModelDefinition, Restore, Dataset, PredictionRecord, CustomDatasetDescription
from assessors.cli.shared import DatasetHub, AssessorHub
from assessors.cli.cli import cli
from assessors.cli.commands.evaluate import evaluate_assessor
from assessors.cli.commands.train_ import TrainArgs


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

    import tensorflow as tf

    n_systems = 5

    def to_supervised(record: PredictionRecord):
        syst_id = tf.one_hot(record['syst_id'], depth=n_systems, dtype=tf.float64)
        inst_weight = 4.0 if record['syst_id'] == 0 else 1.0
        return (record['inst_features'], syst_id), record['syst_pred_score'], inst_weight

    _dataset: Dataset[PredictionRecord, Any] = CustomDatasetDescription(
        path=args.dataset).load_all()
    supervised = _dataset.map(to_supervised)

    path = Path(f"artifacts/assessors/{args.dataset_name}/{args.model}/{args.identifier}/")

    (train, test) = supervised.split_relative(-0.25)
    logging.info(f'Train size: {len(train)}, test size: {len(test)}')
    logging.info(f'Training {model_def.name()}')
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
