from .model import Model

class Sequential(Model):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    """
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Using __setattr__ from Module will automatically register these in self._modules
            setattr(self, str(idx), module)

    def forward(self, x):
        for name, module in self._modules.items():
            if name.isdigit():
                x = module(x)
        return x
