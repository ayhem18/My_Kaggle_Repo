"""
This scripts contains functionalities to manipulate files and directories
"""
import os
from pathlib import Path
from typing import Union, Optional
import zipfile
import shutil
from datetime import datetime
import re

HOME = os.getcwd()


def abs_path(path: Union[str, Path]) -> Path:
    return Path(path) if os.path.isabs(path) else os.path.join(HOME, path)


def squeeze_directory(directory_path: Union[str, Path]) -> None:
    # Given a directory with only one subdirectory, this function moves all the content of
    # subdirectory to the parent directory

    # first convert to abs
    path = abs_path(directory_path)

    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    if len(files) == 1 and os.path.isdir(os.path.join(path, files[0])):
        subdir_path = os.path.join(path, files[0])
        # copy all the files in the subdirectory to the parent one
        for file_name in os.listdir(subdir_path):
            shutil.move(src=os.path.join(subdir_path, file_name), dst=path)
        # done forget to delete the subdirectory
        os.rmdir(subdir_path)


def copy_directories(src_dir: str, des_dir: str, copy: bool = True,
                     filter_directories: callable = None) -> None:
    # convert the src_dir and des_dir to absolute paths
    src_dir, des_dir = abs_path(src_dir), abs_path(des_dir)

    assert os.path.isdir(src_dir) and os.path.isdir(des_dir), "BOTH ARGUMENTS MUST BE DIRECTORIES"

    if filter_directories is None:
        def filter_directories(x):
            return True

    # iterate through each file in the src_dir
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)
        if filter_directories(file_name):
            if copy:
                shutil.copy(file_path, des_dir)
            else:
                shutil.move(file_path, des_dir)

    # remove the source directory if it is currently empty
    if os.listdir(src_dir) == 0:
        os.rmdir(src_dir)


def unzip_data_file(data_zip_path: Union[Path, str],
                    unzip_directory: Optional[Union[Path, str]]=None, 
                    remove_inner_zip_files: bool = True) -> Path:
    data_zip_path = abs_path(data_zip_path)

    assert os.path.exists(data_zip_path), "MAKE SURE THE DATA'S PATH IS SET CORRECTLY!!"

    if unzip_directory is None:
        unzip_directory = Path(data_zip_path).parent

    unzipped_dir = os.path.join(unzip_directory, os.path.basename(os.path.splitext(data_zip_path)[0]))
    os.makedirs(unzipped_dir, exist_ok=True)

    # let's first unzip the file
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        # extract the data to the unzipped_dir
        zip_ref.extractall(unzipped_dir)

    # unzip any files inside the subdirectory
    for file_name in os.listdir(unzipped_dir):
        file_path = os.path.join(unzipped_dir, file_name)
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # extract the data to current directory
                zip_ref.extractall(unzipped_dir)

        # remove the zip files if the flag is set to True
        if remove_inner_zip_files:
            os.remove(file_path)

    # squeeze all the directories
    for file_name in os.listdir(unzipped_dir):
        squeeze_directory(os.path.join(unzipped_dir, file_name))

    return unzipped_dir


DEFAULT_ERROR_MESSAGE = 'MAKE SURE THE passed path satisfies the condition passed with it'

def process_save_path(save_path: Union[str, Path, None],
                      dir_ok: bool = True,
                      file_ok: bool = True,
                      condition: callable = None,
                      error_message: str = DEFAULT_ERROR_MESSAGE) -> Union[str, Path, None]:
    if save_path is not None:
        # first make the save_path absolute
        save_path = save_path if os.path.isabs(save_path) else os.path.join(HOME, save_path)
        assert not \
            ((not file_ok and os.path.isfile(save_path)) or
             (not dir_ok and os.path.isdir(save_path))), \
            f'MAKE SURE NOT TO PASS A {"directory" if not dir_ok else "file"}'

        assert condition is None or condition(save_path), error_message

        # create the directory if needed
        if not os.path.isfile(save_path):
            os.makedirs(save_path, exist_ok=True)

    return save_path


def default_file_name(hour_ok: bool = True,
                        minute_ok: bool = True):
    # Get timestamp of current date (all experiments on certain day live in same folder)
    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format
    timestamp += f"-{(current_hour if hour_ok else '')}-{current_minute if minute_ok else ''}"  # now it is much more detailed: better tracking
    
    # make sure to remove any '-' left at the end
    timestamp = re.sub(r'-+$', '', timestamp)
    return timestamp
