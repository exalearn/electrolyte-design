import contextlib
import os


@contextlib.contextmanager
def working_directory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    Thanks to: http://code.activestate.com/recipes/576620-changedirectory-context-manager/

    Args:
        path (str): Desired working directory
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev_cwd)