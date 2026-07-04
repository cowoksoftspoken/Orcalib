import random

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Normalize:
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        # Apply this to flat lists since DataLoader handles list concatenation
        if isinstance(x, list):
            # Assumes single channel flat image
            if len(self.mean) == 1:
                return [(v - self.mean[0]) / (self.std[0] + 1e-7) for v in x]
        return x

class ToTensor:
    """Convert a list or numpy array to orca.Tensor."""
    def __call__(self, x):
        import orca
        import numpy as np
        if isinstance(x, orca.Tensor):
            return x
        if isinstance(x, list):
            return orca.Tensor.from_list(x, shape=[len(x)])
        if isinstance(x, np.ndarray):
            return orca.Tensor.from_list(x.flatten().tolist(), shape=list(x.shape))
        return x

class RandomCrop:
    """Crop the given image at a random location."""
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        # Implementation depends on image shape, which isn't stored in flat lists.
        # This will be fully implemented when Conv2d and Nd image datasets are added.
        return x

class RandomFlip:
    """Horizontally flip the given image randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            if isinstance(x, list):
                # Naive reverse for 1D lists. Real 2D flip needs shape info.
                return x[::-1]
        return x
