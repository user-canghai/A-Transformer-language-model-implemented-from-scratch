import importlib.metadata

try:
    __version__ = importlib.metadata.version("lm_basics")
except importlib.metadata.PackageNotFoundError:
    pass
