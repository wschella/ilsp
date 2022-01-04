import logging
import os


# We need to do this before importing TF
from assessors.utils.tf_logging import set_tf_loglevel
set_tf_loglevel(logging.WARN)  # nopep8

from tensorflow.python.ops.numpy_ops import np_config
# Used for .flatten() or .reshape() on Tensors
np_config.enable_numpy_behavior()  # nopep8

from assessors.cli import cli

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    datefmt="[%H:%M:%S]",
    format="%(asctime)s:%(levelname)s:%(name)s:%(module)s %(message)s")
logging.getLogger("absl").setLevel(logging.WARNING)


def run():
    cli()


if __name__ == "__main__":
    run()
