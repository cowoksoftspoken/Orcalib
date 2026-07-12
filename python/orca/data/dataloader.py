import random
import orca
from typing import Any, Tuple, Iterator, List, Generic, TypeVar

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co]):
    """
    An abstract class representing a Dataset.
    
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite `__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    `__len__`, which is expected to return the size of the dataset.
    """
    def __len__(self) -> int:
        raise NotImplementedError
        
    def __getitem__(self, idx: int) -> T_co:
        raise NotImplementedError

class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load. Default: 1.
        shuffle (bool, optional): set to True to have the data reshuffled at every epoch. Default: False.
    """
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[Tuple[orca.Tensor, orca.Tensor]]:
        """
        Returns an iterator over the dataset batches.
        
        Yields:
            Tuple[Tensor, Tensor]: A batch of inputs and their corresponding labels.
        """
        def flatten_list(nested_list: Any) -> List[Any]:
            if not isinstance(nested_list, list):
                return [nested_list]
            flat = []
            for item in nested_list:
                flat.extend(flatten_list(item))
            return flat

        def get_shape(nested_list: Any) -> List[int]:
            if not isinstance(nested_list, list):
                return []
            if len(nested_list) == 0:
                return [0]
            return [len(nested_list)] + get_shape(nested_list[0])

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            
            x_samples = [x for x, y in batch]
            y_samples = [y for x, y in batch]
            
            x_shape = [len(batch_indices)] + get_shape(x_samples[0])
            y_shape = [len(batch_indices)] + get_shape(y_samples[0])
            
            X_flat = flatten_list(x_samples)
            Y_flat = flatten_list(y_samples)
            
            X_tensor = orca.Tensor.from_list(X_flat, shape=x_shape)
            Y_tensor = orca.Tensor.from_list(Y_flat, shape=y_shape)
            
            yield X_tensor, Y_tensor
