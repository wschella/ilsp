from typing import *
from dataclasses import dataclass

import click

from assessors.utils.cli import CommandArguments
from assessors.hubs import AssessorHub, DatasetHub
from assessors.application import cli, CLIArgs


@dataclass
class InferInstDifficultyArgs(CommandArguments):
    identifier: str
    inst_ids: List[int]
    parent: CLIArgs = CLIArgs()
    dataset_name: str = "mnist"
    model: str = "default"

    def validate(self):
        self.parent.validate()
        self.validate_option('dataset_name', DatasetHub.options())
        self.validate_option('model', AssessorHub.options_for(self.dataset_name))


@cli.command('infer-inst-difficulty')
@click.argument('inst_ids', type=int, nargs=-1)
@click.option('-i', '--identifier', required=True, help='Identifier for the assessor.')
@click.option('-d', '--dataset_name', required=True, help='Dataset name.')
@click.option('-m', '--model', default="default", help='Model name.')
@click.pass_context
def infer_inst_difficulty(ctx, **kwargs):
    raise NotImplementedError()
#     args = InferInstDifficultyArgs(parent=ctx.obj, **kwargs).validated()

#     # Load assessor model
#     model_def: ModelDefinition = AssessorHub.get(args.dataset_name, args.model)()
#     model_path = Path(
#         f"artifacts/assessors/{args.dataset_name}/{args.model}/{args.identifier}/")
#     model = model_def.restore_from(model_path)

#     ds = DatasetHub.get(args.dataset_name).load_all()
#     df = pd.DataFrame(tfds.as_numpy(ds.ds))  # type: ignore

#     selected = df[df.index.isin(args.inst_ids)]
#     n_systems = 5

#     def to_supervised(syst_id, xy):
#         inst, target = xy
#         one_hot = tf.one_hot(syst_id, depth=n_systems, dtype=tf.float64)
#         return ((inst, one_hot), target)

#     for inst_id in args.inst_ids:
#         row_df = df[df.index == inst_id]
#         target = row_df.pop(1)  # type: ignore
#         target = pd.concat([target] * n_systems)
#         inst = pd.DataFrame(row_df[0].tolist())
#         inst = pd.concat([inst] * n_systems)

#         mini_ds = tf.data.Dataset.from_tensor_slices((inst.values, target))
#         mini_ds = mini_ds.enumerate().map(to_supervised)

#         preds = model.predict_all(TFDataset(mini_ds))
#         print(f'''
# Instance {inst_id}
# avg: {np.mean(preds)}
# std: {np.std(preds)}
# predictions:
#     {preds}
#         ''')
