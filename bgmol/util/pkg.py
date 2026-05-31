import os

from importlib.resources import files, as_file

__all__ = ["get_data_file"]


def get_data_file(relative_path):
    """Get the full path to one of the reference files.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).

    """

    with as_file(files('bgmol.systems') / relative_path) as p:
        fn = str(p)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn