from pp.cli.app import cli

import pp.utils.gpu

@cli.command()
def debug_gpu():
    pp.utils.gpu.debug_gpu()
