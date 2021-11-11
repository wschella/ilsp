from typing import *
from dataclasses import dataclass
from pathlib import Path
import csv
import os

import click
import pandas as pd
from tqdm import tqdm

from assessors.core import ModelDefinition, CustomDatasetDescription, Dataset, PredictionRecord, AssessorPredictionRecord
from assessors.cli.shared import CommandArguments, get_assessor_def
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
        self.validate_option('model', ["mnist_default", "mnist_prob", "cifar10_default"])


@cli.command(name='eval-assessor')
@click.argument('dataset', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-path', default="./results.csv", type=click.Path(), help="The output path")
@click.option('-t', '--test-size', default=10000, help="The test set size")
@click.option('--overwrite', is_flag=True, help="Overwrite the output file if it exists", default=False)
@click.option('-m', '--model', default='mnist_default', help="The model to evaluate")
@click.pass_context
def evaluate_assessor(ctx, **kwargs):
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

        print("Writing results. No idea yet why this is slow.")
        for record, asss_pred in tqdm(zip(test.as_numpy_iterator(), asss_predictions), total=len(test)):
            record: PredictionRecord = record
            record.pop('inst_features')

            record: AssessorPredictionRecord = record
            record['syst_prediction'] = record['syst_prediction'].tolist()
            record['syst_pred_score'] = record['syst_pred_score'][0]
            record["asss_prediction"] = asss_pred.tolist()
            record["asss_pred_loss"] = int(model.loss(record['syst_pred_score'], asss_pred))
            writer.writerow(record)
    print(f"Wrote results to {args.output_path}")

    # Print some simple results
    with open(args.output_path, 'r', newline='') as csvfile:
        df = pd.read_csv(csvfile)
        print(df.describe())
