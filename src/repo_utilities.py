import os
from pathlib import Path


def repo_root():
    # depending on the environment on which the script is run, the HOME variable might differ
    # it can be either at the root of the repository or the current directory of the file
    # this function shall return, the project's root
    h = os.getcwd()
    if 'src' not in os.listdir(h):
        # it means HOME represents the script's parent directory
        while 'src' not in os.listdir(h):
            h = Path(h).parent

    return str(h)

