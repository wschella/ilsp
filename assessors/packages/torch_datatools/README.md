# Torch Datatools

Set of extensions and tools for working with Torch Datasets and DataLoaders until <https://github.com/pytorch/data> is in a stable state.

Some tenets:

- Composable & reusable operations.
  - Simple first, powerful maybe later.
  - Interoperable with other torch dataset extensions
- Giving you no reason to use Tensorflow Datasets for simple functionality, e.g. .skip(), .take(), .interleave()
- Dependencies are all optional, with the exception of torch itself. I'll be looking for a way to depend only on the types, or don't depend on it at all, or depend on it very loosely (such that it doesn't interfere with your specific version).

You should be able to just include this library and use what you need without any overhead. If you need very little, feel free to just copy the relevant methods (with attribution please).

## Scope

- Nothing domain specific. Mostly things that you would expect to find in functools or itertools, but then snugly in the Torch Dataset API.
- We only care about map style datasets, e.g. random-access to an entry by index is always possible.
- Random-acces to an entry is not expensive.
- We assume indexes are integral for a lot of the compositions.
- We assume indexes go from 0 to len(dataset) without holes.

## Potential upcoming features

- Compositions
  - [ ] Interleave
  - [ ] Reference
  - [ ] Shuffle
  - [ ] Unique
- Sources
  - [ ] Numpy ndarray
  - [ ] CSV
  - [ ] Constant
- Transforms
  - [ ] Label to int
  - [ ] Onehot-encode label
  - [ ] Instance weights
- [ ] Wrapper. Makes all the compositions available straight on the dataset.

## Sources

- <https://pytorch.org/docs/stable/data.html>
  Existing already:
  - ConcatDataset (or ChainDataset for Iterable-style)
  - TensorDataset
  - SubsetDataset
- <https://github.com/pytorch/data>
- <https://github.com/szymonmaszke/torchdatasets>
