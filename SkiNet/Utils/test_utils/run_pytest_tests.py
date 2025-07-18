import pytest
import logging

from SkiNet.Utils import project_paths


def run_pytest_tests(run_azure_tests: bool = False):
    """
    Run pytest tests located in project_paths.TESTS_DIR directory

    :param run_azure_tests If True, also runs Azure integration tests. If False, skips them. Default is False
    """
    if run_azure_tests:
        pytest_args = [project_paths.TESTS_DIR, "-v", "-s"]
    else:
        pytest_args = [project_paths.TESTS_DIR, "-v", "-s", "-m", "not azure"]
    
    logging.getLogger(__name__).info("Running pytest using following arguments: %s", pytest_args)
    pytest.main(pytest_args)
