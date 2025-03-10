import pytest
import logging

from SkiNet.Utils import project_paths


def run_pytest_tests():
    """
    Run pytest tests located in project_paths.TESTS_DIR directory
    """
    pytest_args = [project_paths.TESTS_DIR,"-v", "-s"]
    logging.getLogger(__name__).info("Running pytest using following arguments: %s", pytest_args)
    pytest.main(pytest_args)