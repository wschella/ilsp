from typing import *


def setup_logging():
    import logging
    import os

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        datefmt="[%H:%M:%S]",
        format="%(asctime)s:%(levelname)s:%(name)s:%(module)s %(message)s")


def setup_rng(seed: int):
    import numpy as np
    import torch
    import random
    import os

    random.seed(seed)
    torch.manual_seed(random.randint(1, 1_000_000))
    torch.cuda.manual_seed(random.randint(1, 1_000_000))
    np.random.seed(random.randint(1, 1_000_000))
    os.environ["PYTHONHASHSEED"] = str(random.randint(1, 1_000_000))

    # import torch.backends.cudnn
    # torch.backends.cudnn.deterministic = True
