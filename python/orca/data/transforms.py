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
        import numpy as np
        if isinstance(x, np.ndarray):
            mean = np.array(self.mean)
            std = np.array(self.std)
            if x.ndim == 3 and x.shape[0] == len(self.mean):
                mean = mean.reshape(-1, 1, 1)
                std = std.reshape(-1, 1, 1)
            return (x - mean) / (std + 1e-7)
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
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, x):
        import numpy as np
        if isinstance(x, np.ndarray) and x.ndim >= 2:
            h, w = x.shape[-2:]
            th, tw = self.size
            if h < th or w < tw:
                raise ValueError(f"Required crop size {self.size} is larger than input image size {(h, w)}")
            if w == tw and h == th:
                return x
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            if x.ndim == 3:
                return x[:, i:i+th, j:j+tw]
            return x[i:i+th, j:j+tw]
        return x

class RandomFlip:
    """Horizontally flip the given image randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        import numpy as np
        if random.random() < self.p:
            if isinstance(x, np.ndarray) and x.ndim >= 2:
                return np.flip(x, axis=-1).copy()
            if isinstance(x, list):
                # Naive reverse for 1D lists. Real 2D flip needs shape info.
                return x[::-1]
        return x
