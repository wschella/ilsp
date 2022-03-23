# Performance Prediction

Assessor models and the likes.

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
