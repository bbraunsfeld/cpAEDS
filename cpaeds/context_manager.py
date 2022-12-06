from pathlib import Path
import os

class set_directory(object):
    """Sets the cwd within the context

      Args:
          path (Path): The path to the cwd
    """
    def __init__(self, path: Path):
        self.path = path
        self.origin = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.origin)