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

- We only care about map style datasets, e.g. random-access to an entry by index is always possible.
  - This means random-acces to an entry is should not be too expensive.
  - This means some functionality like .filter() or .flat_map() will likely not be available.
- We do not care about domain specific logic.
  We fill focus on functionality things that you would expect to find in functools or itertools, but then snugly in the Torch Dataset API.
- We assume indexes are integral.
  Functionality might work with non-integral indexes, but it is never tested.
- We assume indexes go from 0 to len(dataset) without holes.

## Potential upcoming features

- Compositions
  - [ ] Reexport Torch's ConcatDataset, TensorDataset, and SubsetDataset and random_split
  - [ ] Shuffle
  - [ ] Unique
  - [ ] Cached
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
