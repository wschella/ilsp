# Organisation

## Experiments

We are mainly interested in a failure & score prediction scenario here.

## Functionality

- Base Model Learning
  - Required Assets
    - Base Dataset
  - Required Code
    - Model Definition
  - Produced Assets
    - Learned Base Model
- Base Model Evaluation
  - Required Assets
    - Base Dataset
    - Learned Base Model
- Assessor Training
  - Required Assets
    - Assessor Dataset
  - Required Code
    - Assessor Model Definition
  - Produced Assets
    - Learned Assessor Model
- Assessor Evaluation
  - Required Assets
    - Assessor Dataset
    - Learned Assessor Model
  - Produced Assets
    - Assessor Results
- Assessor Debugging & Analyzing
  - Required Assets
    - Assessor Results
    - Comment: Requires random access to Assessor Results
  - Required Code
    - Manual Result Inspector
      - Jupyter Notebook
    - Result Statistics
      - Comment: Ideally we do this with Pandas or Numpy

## Experiment Parameters

Ideally we want data on every possible combination.

- Assessor dataset
  - Base dataset this was based on
  - (set) of base models
    - parameters (epochs, layers, type)
  - how these are combined with respect to base dataset
- Assessor model
  - parameters

## Assets

- Learned Models
  - Base Model
- Datasets
  - Base Datasets
  - Assessor Datasets
    - Metadata
      - Base dataset this was made from
      - Base models this was made from
      - How this dataset was made (kfold, all same testset)
  - Assessor Results
