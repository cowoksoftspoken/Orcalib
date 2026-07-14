import random
import orca
from typing import Any, Tuple, Iterator, List, Generic, TypeVar, Optional, Union

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


class ArrayDataset(Dataset[Tuple[List[float], List[float]]]):
    """
    High-level Dataset wrapper for in-memory arrays (lists, numpy arrays, or orca.Tensors).
    Supports optional automatic one-hot encoding for target labels.
    """
    def __init__(self, *arrays: Any, one_hot_classes: Optional[int] = None):
        # Convert inputs to nested lists of floats for clean retrieval on CPU
        self.arrays = []
        for arr in arrays:
            if hasattr(arr, 'to_list'):  # orca.Tensor
                flat = arr.to_list()
                shape = arr.shape
                if len(shape) == 1:
                    self.arrays.append([[v] for v in flat])
                elif len(shape) == 2:
                    w = shape[1]
                    self.arrays.append([flat[i * w:(i + 1) * w] for i in range(shape[0])])
                else:
                    # Higher rank tensor slices: flatten slices
                    h_size = 1
                    for d in shape[1:]:
                        h_size *= d
                    self.arrays.append([flat[i * h_size:(i + 1) * h_size] for i in range(shape[0])])
            elif hasattr(arr, 'tolist'):  # numpy array
                lst = arr.tolist()
                # Ensure 1D numpy array target gets parsed as a list of lists if needed
                if isinstance(lst, list) and len(lst) > 0 and not isinstance(lst[0], list):
                    lst = [[v] for v in lst]
                self.arrays.append(lst)
            else:
                lst = list(arr)
                if isinstance(lst, list) and len(lst) > 0 and not isinstance(lst[0], (list, tuple)):
                    lst = [[v] for v in lst]
                self.arrays.append(lst)
                
        if not self.arrays:
            raise ValueError("At least one array must be provided to ArrayDataset")
            
        self.length = len(self.arrays[0])
        for idx, arr in enumerate(self.arrays):
            if len(arr) != self.length:
                raise ValueError(f"Array at index {idx} has length {len(arr)}, expected {self.length}")
                
        # Optional automatic one-hot encoding for targets (assumed to be the last array)
        if one_hot_classes is not None:
            targets = self.arrays[-1]
            one_hot = []
            for t in targets:
                val = int(t[0] if isinstance(t, (list, tuple)) else t)
                if val < 0 or val >= one_hot_classes:
                    raise ValueError(f"Class label {val} is out of bounds for one_hot_classes={one_hot_classes}")
                vec = [0.0] * one_hot_classes
                vec[val] = 1.0
                one_hot.append(vec)
            self.arrays[-1] = one_hot

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[List[float], ...]:
        return tuple(arr[idx] for arr in self.arrays)


class CSVDataset(Dataset[Tuple[List[float], List[float]]]):
    """
    High-level Dataset wrapper for loading tabular data directly from a CSV file.
    """
    def __init__(
        self,
        filepath: str,
        feature_cols: List[Union[int, str]],
        target_cols: List[Union[int, str]],
        has_header: bool = True,
        one_hot_classes: Optional[int] = None
    ):
        import csv
        features = []
        targets = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if not rows:
            raise ValueError(f"CSV file {filepath} is empty")
            
        header = None
        if has_header:
            header = rows[0]
            data_rows = rows[1:]
        else:
            data_rows = rows
            
        def get_indices(cols):
            indices = []
            for col in cols:
                if isinstance(col, str):
                    if not has_header or header is None:
                        raise ValueError("Cannot select column by name without a header row")
                    indices.append(header.index(col))
                else:
                    indices.append(col)
            return indices
            
        feat_indices = get_indices(feature_cols)
        targ_indices = get_indices(target_cols)
        
        for row in data_rows:
            feat_val = [float(row[idx]) for idx in feat_indices]
            targ_val = [float(row[idx]) for idx in targ_indices]
            features.append(feat_val)
            targets.append(targ_val)
            
        self.dataset = ArrayDataset(features, targets, one_hot_classes=one_hot_classes)
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Tuple[List[float], List[float]]:
        return self.dataset[idx]


class SubsetDataset(Dataset[T_co]):
    """
    Dataset wrapping a subset of another dataset.
    """
    def __init__(self, dataset: Dataset[T_co], indices: List[int]):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> T_co:
        return self.dataset[self.indices[idx]]


def random_split(dataset: Dataset[T_co], lengths: List[int]) -> List[SubsetDataset[T_co]]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset")
        
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    subsets = []
    start = 0
    for length in lengths:
        end = start + length
        subsets.append(SubsetDataset(dataset, indices[start:end]))
        start = end
    return subsets

