import random
import orca

class Dataset:
    """
    An abstract class representing a dataset.
    All datasets that represent a map from keys to data samples should subclass it.
    """
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError

class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            
            X_batch = []
            Y_batch = []
            for x, y in batch:
                X_batch.extend(x)
                Y_batch.extend(y)
                
            features_x = len(batch[0][0])
            features_y = len(batch[0][1])
            
            X_tensor = orca.Tensor.from_list(X_batch, shape=[len(batch_indices), features_x])
            Y_tensor = orca.Tensor.from_list(Y_batch, shape=[len(batch_indices), features_y])
            
            yield X_tensor, Y_tensor
