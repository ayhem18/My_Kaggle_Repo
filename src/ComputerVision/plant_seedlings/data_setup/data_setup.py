"""
This script is written to prepare the plant seedlings dataset for modeling.
"""
import os
import re
import sys
import shutil

HOME = os.getcwd()
# sys.path.append(HOME)
# sys.path.append(os.path.join(HOME, 'src'))

from pathlib import Path
from typing import Union

try:
    from src.pytorch_modular.directories_and_files import unzip_data_file, dataset_portion
    from src.repo_utilities import repo_root
except ModuleNotFoundError:
    # the idea here is to apply
    h = os.getcwd()
    if 'src' not in os.listdir(h):
        # it means HOME represents the script's parent directory
        while 'src' not in os.listdir(h):
            h = Path(h).parent
    sys.path.append(str(h))
    sys.path.append(os.path.join(h, 'src'))
    print(sys.path)

from src.pytorch_modular.directories_and_files import unzip_data_file, dataset_portion
from src.repo_utilities import repo_root


def prepare_data(zip_file_path: Union[str, Path] = None,
                 unzip_directory: Union[str, Path] = None):
    home = repo_root()
    if zip_file_path is None:
        # this is the default location of the file
        zip_file_path = os.path.join(home, 'src', 'ComputerVision', 'plant_seedlings', 'data.zip')

    directory = unzip_data_file(zip_file_path, unzip_directory=unzip_directory)
    # to avoid any issues with the file system, let's standardize the names of the classes
    # in the train directory
    # remove any non directories in the folder
    for d in os.listdir(directory):
        d = os.path.join(directory, d)
        if not os.path.isdir(d):
            os.remove(d)

    src_dirs = os.listdir(os.path.join(directory, 'train'))

    for dir_name in src_dirs:
        new_dir_name = re.sub(r'[^a-zA-Z]+', '_', dir_name)
        src = os.path.join(directory, 'train', dir_name)
        des = os.path.join(directory, 'train', new_dir_name)
        if src != des:
            try:
                os.rename(src, des)
            except (FileExistsError, OSError):
                pass
            # remove the old name
            try:

                shutil.rmtree(src)
            except FileNotFoundError:
                pass

        # the names of the original images are impractical (Apparently hashes)
        # let's replace them with better names
        for index, name in enumerate(os.listdir(des)):
            # extract the extension
            _, ext = os.path.splitext(name)
            try:
                os.rename(os.path.join(des, name), os.path.join(des, f'{index}{ext}'))
            except FileExistsError:
                pass
    # remove any non directories


def sanity_check_data(data_directory: Union[str, Path] = None,
                      des_directory: Union[str, Path] = None,
                      portion: Union[str, int] = 0.1):
    home = repo_root()

    data_path = os.path.join(home, 'src', 'ComputerVision', 'plant_seedlings', 'data', 'train') \
        if data_directory is None else data_directory

    dataset_portion(data_path,
                    destination_directory=des_directory,
                    portion=portion)


if __name__ == '__main__':
    sanity_check_data()
