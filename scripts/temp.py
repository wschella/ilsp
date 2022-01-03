import torch
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset


class ReferenceDataset(TorchDataset):
    def __init__(self, source, references):
        self.source = source
        self.references = references

    def __len__(self):
        return len(self.references)

    def __getitem__(self, idx):
        ref = self.references[idx]
        print(ref)
        return self.source[ref]


mnist = tv.datasets.MNIST(
    "datasets/mnist",
    train=True, download=True,
    transform=tv.transforms.Compose([tv.transforms.PILToTensor()]))

indices = np.random.randint(0, len(mnist), size=len(mnist))
refds = ReferenceDataset(mnist, indices)
loader = DataLoader(refds, batch_size=32, shuffle=True)

print(len(refds))
for sample in loader:
    pass
