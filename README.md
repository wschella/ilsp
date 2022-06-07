# Assessor models & and Performance Prediction (PP)

## Running

The entry point to the experiments is "assessors/main.py". Use it by running `poe run` (after setup is done of course). Check the available options with `poe run --help`.
Another option is invoking Python directly `python -m assessors.main --help`


## Organisation

TODO
Dataset and Model wrappers (their interfaces), Hubs, CLI, etc.

## Setup

We use a combination [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for environment management (Python version, CUDA support, ...), and [Poetry](https://python-poetry.org/) for package & dependency management. You only need Conda to get started, which will take care of installing Poetry for you. See [environment.yml](./environment.yml) for details.

```shell
# Create the required Conda environment
conda env create --prefix .env -f environment.yml

# and activate it. This is needed every time when you have a new shell or deactivate it.
conda activate ./.env

# Install all required Python dependencies
poetry install
```

Other useful commands:

```shell
# Add Python dependencies, you can also edit pyproject.toml
poetry add pandas=^1.3

# If you ever need to update the conda env. `--prune` isn't actually doing anything.
conda env update --prefix ./env --file environment.yml  --prune

# Standard deactivating of conda environments
conda deactivate

# Running PoeThePoet scripts (see pyproject.toml) for more info.
poe jpt # Spawns a Jupyterlab server
```
