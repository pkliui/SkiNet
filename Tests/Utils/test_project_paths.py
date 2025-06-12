import pytest
from pathlib import Path
from unittest.mock import patch
from SkiNet.Utils.project_paths import get_repo_root_directory

def test_get_repo_root_directory():
    mocked_file_path = Path('/home/user/project_root/SkiNet/Utils/mocked_file.py')
    expected_root = Path('/home/user/project_root')

    with patch('SkiNet.Utils.project_paths.__file__', str(mocked_file_path)):
        repo_root = get_repo_root_directory()

    # Assert that the returned path matches the expected path
    assert repo_root == expected_root, f"Expected {expected_root}, but got {repo_root}"
