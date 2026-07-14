try:
    import safetensors
    from .loader import load
except ImportError:
    def load(*args, **kwargs):
        raise ImportError(
            "The 'safetensors' library is required to use Hugging Face weight loading features. "
            "Please install it using 'pip install safetensors'."
        )

__all__ = ["load"]
