try:
    import onnx
    from .exporter import export_onnx
    from .importer import import_onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

    def export_onnx(*args, **kwargs):
        raise ImportError(
            "The 'onnx' library is required to use ONNX export features. "
            "Please install it using 'pip install onnx'."
        )

    def import_onnx(*args, **kwargs):
        raise ImportError(
            "The 'onnx' library is required to use ONNX import features. "
            "Please install it using 'pip install onnx'."
        )

__all__ = ["export_onnx", "import_onnx"]
