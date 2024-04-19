import os
import shutil
from pathlib import Path
from typing import Iterable


class FileExistsException(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path


def create_dir_if_not_exist(path: str) -> None:
    """creates dir path if doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def throw_or_remove_file(paths: Iterable[str], force_remove: bool):
    """
    Check whether paths exists. If force_remove is true, existing files are
    removed otherwise if there exists any file, exception is thrown.

    Args:
        paths (Iterable[str]): List of paths.
        force_remove (bool): Whether to remove file if exists or not.

    Raises:
        FileExistsError: When force_remove is False and some file does not
            exists.
    """
    for path in paths:
        if os.path.exists(path):
            if not force_remove:
                raise FileExistsException(path)

            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
