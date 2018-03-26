from torch.utils.data.dataset import Subset
from torch._utils import _accumulate
from torch import randperm
from torchvision import datasets, models, transforms
class MySubset(datasets.ImageFolder):
    def __init__(self, dataset, indices):
        super(MySubset,self).__init__(dataset.root, dataset.transform, dataset.target_transform,dataset.loader)
        self.dataset=dataset
        self.indices=indices
    def __getitem__(self, index):
        return super(MySubset, self).__getitem__(self.indices[index])
    def __len__(self):
        return len(self.indices)

def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [MySubset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]