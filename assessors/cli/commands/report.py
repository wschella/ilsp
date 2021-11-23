from typing import *
from dataclasses import dataclass
from pathlib import Path
import os
import json

import click
import pandas as pd
import numpy as np
import sklearn.metrics as metrics

from assessors.cli.shared import CommandArguments
from assessors.cli.cli import cli, CLIArgs
import assessors.report as rr


@dataclass
class ReportArgs(CommandArguments):
    parent: CLIArgs = CLIArgs()
    results: Path = Path('./results.csv')
    output_path: Path = Path("./results.html")
    overwrite: bool = True

    def validate(self):
        self.parent.validate()


@cli.command(name='report')
@click.argument('results', required=False, default=Path("./results.csv"), type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output-path', default=Path("./results.html"), type=click.Path(path_type=Path), help="The output path")
@click.option('--overwrite/--no-overwrite', is_flag=True, help="Overwrite the output file if it exists", default=True)
@click.pass_context
def generate_report(ctx, **kwargs):
    """
    Generate a report from a results file
    """

    # Handle CLI args
    args = ReportArgs(parent=ctx.obj, **kwargs).validated()
    if os.path.exists(args.output_path) and not args.overwrite:
        click.confirm(f"The file {args.output_path} already exists. Overwrite?", abort=True)

    # Print some simple results and create a report
    with open(args.results, 'r', newline='') as csvfile:

        df = pd.read_csv(csvfile)
        df.asss_prediction = df.asss_prediction.map(lambda s: np.array(json.loads(s)))
        df.syst_prediction = df.syst_prediction.map(lambda s: np.array(json.loads(s)))
        y_score = df['asss_prediction'].map(lambda p: p[0])
        y_pred = y_score.map(lambda p: p > 0.5)
        y_true = df['syst_pred_score']

        print(metrics.classification_report(y_true, y_pred))
        print(metrics.confusion_matrix(y_true, y_pred))
        print(f"AUC: {metrics.roc_auc_score(y_true, y_pred)}")

        # And create a report
        rr.AssessorReport(df).save(Path(args.output_path).with_suffix(".html"))
        print(f"Report saved to {args.output_path}")
