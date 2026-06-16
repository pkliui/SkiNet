import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `import SkiNet` works when this
# script is invoked directly (e.g. via the pre-commit hook) without PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from SkiNet.Utils import project_paths


def run_pytest_tests() -> None:
    """
    Run pytest tests located in project_paths.TESTS_DIR directory
    """
    pytest_args = [str(project_paths.TESTS_DIR), "-v", "-s"]

    logging.getLogger(__name__).info("Running pytest using following arguments: %s", pytest_args)
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run_pytest_tests()
