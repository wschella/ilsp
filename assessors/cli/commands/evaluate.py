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
from assessors.cli.commands.report import generate_report

from assessors.core import ModelDefinition, CustomDatasetDescription, Dataset, PredictionRecord, TypedPredictionRecord, AssessorPredictionRecord, LeanPredictionRecord
from assessors.cli.shared import CommandArguments, AssessorHub, SystemHub, DatasetHub
from assessors.cli.cli import cli, CLIArgs
import assessors.report as rr


@dataclass
class EvaluateAssessorArgs(CommandArguments):
    dataset: Path
    dataset_name: str = "mnist"
    parent: CLIArgs = CLIArgs()
    output_path: Path = Path("./results.csv")
    identifier: str = "f5_r1"
    overwrite: bool = False
    write_system_results: bool = False
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset_name', DatasetHub.options())
        self.validate_option('model', AssessorHub.options_for(self.dataset_name))


@cli.command(name='eval-assessor')
@click.argument('dataset', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('-d', '--dataset-name', required=True, help="The name of the source dataset")
@click.option('-o', '--output-path', default="./results.csv", type=click.Path(path_type=Path), help="The output path")
@click.option('-i', '--identifier', required=True, help="The identifier of the assessor")
@click.option('--overwrite', is_flag=True, help="Overwrite the output file if it exists", default=False)
@click.option('--write-system-results', is_flag=True, help="Also write the system results to the output file (without inst_features)", default=False)
@click.option('-m', '--model', default='default', help="The model to evaluate")
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
    model_def: ModelDefinition = AssessorHub.get(args.dataset_name, args.model)()
    model_path = Path(
        f"artifacts/assessors/{args.dataset_name}/{args.model}/{args.identifier}/")
    model = model_def.restore_from(model_path)

    # Load & mangle dataset
    _ds: Dataset[PredictionRecord, Any] = CustomDatasetDescription(path=args.dataset).load_all()
    (_train, test) = _ds.split_relative(-0.2)

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
        for _record, asss_pred in tqdm(zip(test_as_numpy, asss_predictions), total=len(test)):
            record: TypedPredictionRecord = _record
            asss_record = AssessorPredictionRecord(
                inst_index=record['inst_index'],
                inst_target=record['inst_target'],
                syst_id=record['syst_id'],
                syst_features=record['syst_features'].tolist(),
                syst_prediction=record['syst_prediction'].tolist(),
                syst_pred_score=record['syst_pred_score'],
                syst_pred_loss=record['syst_pred_loss'],
                asss_prediction=asss_pred.tolist(),
                asss_pred_loss=int(model.loss(record['syst_pred_score'], asss_pred))
            )
            writer.writerow(asss_record)
    print(f"Wrote results to {args.output_path}")

    if args.write_system_results:
        with open(args.output_path.with_name("system_results.csv"), "w") as f:
            fieldnames = list(LeanPredictionRecord.__annotations__.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            train_as_numpy: Sequence[TypedPredictionRecord] = _train.as_numpy_sequence()
            print("Writing _system_ results")
            for _record in tqdm(train_as_numpy, total=len(test)):
                record: TypedPredictionRecord = _record
                record.pop('inst_features')  # Make it a LeanPredictionRecord
                record['syst_features'] = record['syst_features'].tolist()
                record['syst_prediction'] = record['syst_prediction'].tolist()
                writer.writerow(record)

    # Print some simple results
    ctx.invoke(
        generate_report,
        results=args.output_path,
        output_path=args.output_path.with_suffix(".html"),
        overwrite=args.overwrite)


# --------------------


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
