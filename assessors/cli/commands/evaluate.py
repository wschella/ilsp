from typing import *
from dataclasses import dataclass
from pathlib import Path
import csv
import os
import json

import click
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

from assessors.core import ModelDefinition, CustomDatasetDescription, Dataset, PredictionRecord, TypedPredictionRecord, AssessorPredictionRecord
from assessors.cli.shared import CommandArguments, get_assessor_def, get_dataset_description, get_model_def
from assessors.cli.cli import cli, CLIArgs


@dataclass
class EvaluateAssessorArgs(CommandArguments):
    dataset: Path
    parent: CLIArgs = CLIArgs()
    test_size: int = 10000
    output_path: Path = Path("./results.csv")
    overwrite: bool = False
    model: str = "mnist_default"

    def validate(self):
        self.parent.validate()
        self.validate_option('model', ["mnist_default", "mnist_prob",
                             "cifar10_default", "segment_default"])


@cli.command(name='eval-assessor')
@click.argument('dataset', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('-o', '--output-path', default="./results.csv", type=click.Path(path_type=Path), help="The output path")
@click.option('-t', '--test-size', default=10000, help="The test set size")
@click.option('--overwrite', is_flag=True, help="Overwrite the output file if it exists", default=False)
@click.option('-m', '--model', default='mnist_default', help="The model to evaluate")
@click.pass_context
def evaluate_assessor(ctx, **kwargs):
    """
    Evaluate an assessor, currently only really works for predicting binary scores.
    """

    # Handle CLI args
    args = EvaluateAssessorArgs(parent=ctx.obj, **kwargs).validated()
    if os.path.exists(args.output_path) and not args.overwrite:
        click.confirm(f"The file {args.output_path} already exists. Overwrite?", abort=True)

    # Load assessor model
    [dataset_name, model_name] = args.model.split('_')
    model_def: ModelDefinition = get_assessor_def(dataset_name, model_name)()
    model_path = Path(f"artifacts/models/{dataset_name}/{model_name}/assessor/")
    model = model_def.restore_from(model_path)

    # Load & mangle dataset
    _ds: Dataset[PredictionRecord, Any] = CustomDatasetDescription(path=args.dataset).load_all()
    (_train, test) = _ds.split_absolute(-args.test_size)

    def to_supervised(record: PredictionRecord):
        return (record['inst_features'], record['syst_pred_score'])

    # Evaluate and log
    os.makedirs(os.path.dirname(args.output_path.resolve()), exist_ok=True)
    with open(args.output_path, "w") as f:
        fieldnames = list(AssessorPredictionRecord.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        asss_predictions = model.predict_all(test.map(to_supervised))
        test_as_numpy: Sequence[TypedPredictionRecord] = test.as_numpy_sequence()

        print("Writing results. No idea yet why this is slow.")
        for record, asss_pred in tqdm(zip(test_as_numpy, asss_predictions), total=len(test)):
            asss_record = AssessorPredictionRecord(
                inst_index=record['inst_index'],
                inst_target=record['inst_target'],
                syst_features=record['syst_features'].tolist(),
                syst_prediction=record['syst_prediction'].tolist(),
                syst_pred_score=record['syst_pred_score'],
                syst_pred_loss=record['syst_pred_loss'],
                asss_prediction=asss_pred.tolist(),
                asss_pred_loss=int(model.loss(record['syst_pred_score'], asss_pred))
            )
            writer.writerow(asss_record)
    print(f"Wrote results to {args.output_path}")

    # Print some simple results
    with open(args.output_path, 'r', newline='') as csvfile:

        df = pd.read_csv(csvfile)
        df.asss_prediction = df.asss_prediction.map(lambda s: np.array(json.loads(s)))
        y_score = df['asss_prediction'].map(lambda p: p[0])
        y_pred = y_score.map(lambda p: p > 0.5)
        y_true = df['syst_pred_score']

        print(metrics.classification_report(y_true, y_pred))
        print(metrics.confusion_matrix(y_true, y_pred))
        print(f"AUC: {metrics.roc_auc_score(y_true, y_pred)}")

# --------------------


@dataclass
class EvaluateSystemArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    dataset: str = "mnist"
    test_size: int = 10000
    model: str = "default"

    def validate(self):
        self.parent.validate()


@cli.command(name='eval-system')
@click.argument('dataset')
@click.option('-t', '--test-size', default=10000, help="The test set size")
@click.option('-m', '--model', default='default', help="The model to evaluate")
@click.pass_context
def evaluate_system(ctx, **kwargs):
    # Handle CLI args
    args = EvaluateSystemArgs(parent=ctx.obj, **kwargs).validated()

    # Load system
    model_def: ModelDefinition = get_model_def(args.dataset, args.model)()
    # TODO: Fix
    model_path = Path(f"artifacts/models/{args.dataset}/{args.model}/kfold_5/4")
    model = model_def.restore_from(model_path)

    # Load & mangle dataset
    _ds: Dataset = get_dataset_description(args.dataset).load_all()
    (_train, test) = _ds.split_absolute(-args.test_size)

    y_true = test.map(lambda e: e[1]).as_numpy()
    y_pred = model.predict_all(test.map(lambda e: e[0]))
    y_pred = np.argmax(y_pred, axis=1)

    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))
