import os
from typing import Union
from pathlib import Path
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_or_create_out_dir(out_dir_path: Path) -> Path:
    if os.path.exists(out_dir_path):
        if len(os.listdir(out_dir_path)) != 0:
            logger.warning(f"Directory {out_dir_path} already exists and contains files")
            raise FileExistsError(f"Directory {out_dir_path} already exists and contains files")
    else:
        os.mkdir(out_dir_path)
    return out_dir_path


def remove_files_in_dir(curr_run_dir: Union[Path, str], suffix: str = None) -> None:
    for f in os.listdir(curr_run_dir):
        if suffix and suffix in f:
            os.remove(curr_run_dir.joinpath(f))
