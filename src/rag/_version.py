"""Version helpers."""

from importlib.metadata import PackageNotFoundError, version

__version__ = "0.1.0"


def get_version() -> str:
    """Return the installed package version when available."""

    try:
        return version("python-rag-framework")
    except PackageNotFoundError:
        return __version__
